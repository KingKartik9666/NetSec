//! Numerical port of XGBoost v3.1.3's exact, path-dependent TreeSHAP.
//! Copyright 2017-2025 XGBoost Contributors. Apache-2.0; see NOTICE.
//! Conditioning/interactions are intentionally outside this attribution spike.
use crate::{Error, Tree};

#[derive(Clone, Copy, Default)]
struct Element {
    feature: Option<usize>,
    zero: f32,
    one: f32,
    weight: f32,
}

fn extend(path: &mut [Element], depth: usize, zero: f32, one: f32, feature: Option<usize>) {
    path[depth] = Element {
        feature,
        zero,
        one,
        weight: if depth == 0 { 1.0 } else { 0.0 },
    };
    for i in (0..depth).rev() {
        path[i + 1].weight += one * path[i].weight * (i + 1) as f32 / (depth + 1) as f32;
        path[i].weight = zero * path[i].weight * (depth - i) as f32 / (depth + 1) as f32;
    }
}

fn unwind(path: &mut [Element], depth: usize, index: usize) {
    let one = path[index].one;
    let zero = path[index].zero;
    let mut next = path[depth].weight;
    for i in (0..depth).rev() {
        if one != 0.0 {
            let old = path[i].weight;
            path[i].weight = next * (depth + 1) as f32 / ((i + 1) as f32 * one);
            next = old - path[i].weight * zero * (depth - i) as f32 / (depth + 1) as f32;
        } else {
            path[i].weight = path[i].weight * (depth + 1) as f32 / (zero * (depth - i) as f32);
        }
    }
    for i in index..depth {
        path[i].feature = path[i + 1].feature;
        path[i].zero = path[i + 1].zero;
        path[i].one = path[i + 1].one;
    }
}

fn unwound_sum(path: &[Element], depth: usize, index: usize) -> f32 {
    let one = path[index].one;
    let zero = path[index].zero;
    let mut next = path[depth].weight;
    let mut total = 0.0;
    for i in (0..depth).rev() {
        if one != 0.0 {
            let value = next * (depth + 1) as f32 / ((i + 1) as f32 * one);
            total += value;
            next = path[i].weight - value * zero * ((depth - i) as f32 / (depth + 1) as f32);
        } else if zero != 0.0 {
            total += (path[i].weight / zero) / ((depth - i) as f32 / (depth + 1) as f32);
        }
    }
    total
}

struct Branch {
    node: usize,
    depth: usize,
    zero: f32,
    one: f32,
    feature: Option<usize>,
}

fn visit(
    tree: &Tree,
    row: &[Option<f32>],
    phi: &mut [f32],
    parent: &[Element],
    branch: Branch,
) -> Result<(), Error> {
    let Branch {
        node: index,
        mut depth,
        zero,
        one,
        feature,
    } = branch;
    let node = &tree.nodes[index];
    let mut path = vec![Element::default(); depth + 2];
    for (target, source) in path.iter_mut().zip(parent).take(depth + 1) {
        *target = *source;
    }
    extend(&mut path, depth, zero, one, feature);
    if let (Some(left), Some(right)) = (node.left, node.right) {
        let hot = if row[node.feature].map_or(node.default_left, |v| v < node.value) {
            left
        } else {
            right
        };
        let cold = if hot == left { right } else { left };
        let hot_zero = tree.nodes[hot].cover / node.cover;
        let cold_zero = tree.nodes[cold].cover / node.cover;
        let mut incoming_zero = 1.0;
        let mut incoming_one = 1.0;
        if let Some(previous) = path[..=depth]
            .iter()
            .position(|item| item.feature == Some(node.feature))
        {
            incoming_zero = path[previous].zero;
            incoming_one = path[previous].one;
            unwind(&mut path, depth, previous);
            depth = depth
                .checked_sub(1)
                .ok_or_else(|| Error::Invalid("Invalid repeated TreeSHAP path".into()))?;
        }
        visit(
            tree,
            row,
            phi,
            &path,
            Branch {
                node: hot,
                depth: depth + 1,
                zero: hot_zero * incoming_zero,
                one: incoming_one,
                feature: Some(node.feature),
            },
        )?;
        visit(
            tree,
            row,
            phi,
            &path,
            Branch {
                node: cold,
                depth: depth + 1,
                zero: cold_zero * incoming_zero,
                one: 0.0,
                feature: Some(node.feature),
            },
        )?;
    } else {
        for i in 1..=depth {
            let feature = path[i]
                .feature
                .ok_or_else(|| Error::Invalid("TreeSHAP feature index missing".into()))?;
            phi[feature] +=
                unwound_sum(&path, depth, i) * (path[i].one - path[i].zero) * node.value;
        }
    }
    Ok(())
}

pub(crate) fn explain(
    tree: &Tree,
    row: &[Option<f32>],
    features: usize,
) -> Result<Vec<f32>, Error> {
    let mut phi = vec![0.0; features + 1];
    phi[features] = tree.mean;
    visit(
        tree,
        row,
        &mut phi,
        &[Element::default()],
        Branch {
            node: 0,
            depth: 0,
            zero: 1.0,
            one: 1.0,
            feature: None,
        },
    )?;
    if phi.iter().any(|v| !v.is_finite()) {
        return Err(Error::Invalid("Nonfinite TreeSHAP contributions".into()));
    }
    Ok(phi)
}
