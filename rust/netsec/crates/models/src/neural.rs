use crate::{invalid,ModelError,tensor::Tensors};
use serde_json::Value;

fn integer(v:&Value)->Result<usize,ModelError>{v.as_u64().and_then(|v|usize::try_from(v).ok()).filter(|v|*v<=8192).ok_or_else(||invalid("Invalid network dimension"))}
pub fn dot(a:&[f32],b:&[f32])->f32{a.iter().zip(b).map(|(a,b)|*a as f64 * *b as f64).sum::<f64>() as f32}
pub fn softmax(x:&[f32])->Vec<f64>{
    let largest=x.iter().copied().fold(f32::NEG_INFINITY,f32::max);
    let scores:Vec<f32>=x.iter().map(|v|(*v-largest).exp()).collect();
    let total=scores.iter().map(|v|*v as f64).sum::<f64>();
    scores.iter().map(|v|((*v as f64/total) as f32) as f64).collect()
}
pub fn sigmoid(v:f64)->f64{if v>=0.0{1.0/(1.0+(-v).exp())}else{let e=v.exp();e/(1.0+e)}}

pub fn bspline(x:f32,grid:&[f32],order:usize)->Vec<f32>{
    let mut bases:Vec<f32>=grid.windows(2).map(|g|if x>=g[0]&&x<g[1]{1.0}else{0.0}).collect();
    for k in 1..=order{
        bases=(0..bases.len()-1).map(|i|{
            ((x-grid[i])/(grid[i+k]-grid[i]))*bases[i]+
            ((grid[i+k+1]-x)/(grid[i+k+1]-grid[i+1]))*bases[i+1]
        }).collect();
    }
    bases
}

pub fn kan(manifest:&Value,tensors:&Tensors,mut x:Vec<f32>)->Result<Vec<f64>,ModelError>{
    let mut widths:Vec<usize>=manifest["config"]["hidden"].as_array().ok_or_else(||invalid("Missing KAN widths"))?
        .iter().map(integer).collect::<Result<_,_>>()?;
    widths.push(manifest["classes"].as_array().ok_or_else(||invalid("Missing classes"))?.len());
    let order=integer(&manifest["config"]["spline_order"])?;
    let intervals=integer(&manifest["config"]["grid_size"])?;
    if order>8 || widths.len()>16 || intervals>128{return Err(invalid("KAN architecture outside bounds"))}
    let basis_count=intervals+order;let grid_count=intervals+2*order+1;
    for (layer,&out) in widths.iter().enumerate(){
        let prefix=format!("kan.layers.{layer}");
        let base=tensors.f32(&format!("{prefix}.base_weight"))?;
        let spline=tensors.f32(&format!("{prefix}.spline_weight"))?;
        let scaler=tensors.f32(&format!("{prefix}.spline_scaler"))?;
        let grid=tensors.f32(&format!("{prefix}.grid"))?;
        let input=x.len();
        if input==0 || base.len()!=out*input || spline.len()!=out*input*basis_count || scaler.len()!=out*input || grid.len()!=input*grid_count{
            return Err(invalid("KAN tensor shape differs"))
        }
        let activated:Vec<f32>=x.iter().map(|v|*v*sigmoid(*v as f64) as f32).collect();
        let mut encoded=Vec::with_capacity(input*basis_count);
        for i in 0..input{
            let g=&grid[i*grid_count..(i+1)*grid_count];
            if g.windows(2).any(|p|p[1]<=p[0]){return Err(invalid("KAN grids must be increasing"))}
            encoded.extend(bspline(x[i],g,order));
        }
        x=(0..out).map(|o|{
            let base_output=dot(&activated,&base[o*input..(o+1)*input]);
            let mut spline_sum=0.0f64;
            for i in 0..input{for b in 0..basis_count{
                let index=(o*input+i)*basis_count+b;
                let scaled=spline[index]*scaler[o*input+i];
                spline_sum+=scaled as f64*encoded[i*basis_count+b] as f64;
            }}
            base_output+spline_sum as f32
        }).collect();
    }
    Ok(softmax(&x))
}

pub fn mlp(tensors:&Tensors,prefix:&str,input:&[f32],widths:&[usize])->Result<Vec<f32>,ModelError>{
    let mut x=input.to_vec();
    for (layer,&out) in widths.iter().enumerate(){
        let name=format!("{prefix}.model.{}",2*layer);
        let w=tensors.f32(&format!("{name}.weight"))?;
        let b=tensors.f32(&format!("{name}.bias"))?;
        if x.is_empty()||w.len()!=out*x.len()||b.len()!=out{return Err(invalid("MLP tensor shape differs"))}
        x=(0..out).map(|o|{
            let value=dot(&x,&w[o*x.len()..(o+1)*x.len()])+b[o];
            if layer+1<widths.len(){value.max(0.0)}else{value}
        }).collect();
    }Ok(x)
}

fn embedding(manifest:&Value,tensors:&Tensors,x:&[f32])->Result<Vec<f32>,ModelError>{
    if manifest["embedded"].as_bool()!=Some(true){return Ok(x.to_vec())}
    let numeric=tensors.i32("numeric_indices")?;let bypass=tensors.i32("bypass_indices")?;
    let weight0=tensors.f32("embeddings.linear0.weight")?;let bias0=tensors.f32("embeddings.linear0.bias")?;
    let encoding_w=tensors.f32("embeddings.impl.weight")?;let encoding_b=tensors.f32("embeddings.impl.bias")?;
    let embedding_w=tensors.f32("embeddings.linear.weight")?;
    let d=integer(&manifest["config"]["d_embedding"])?;
    if numeric.is_empty()||d==0||encoding_w.len()%numeric.len()!=0{return Err(invalid("Invalid embedding dimensions"))}
    let bins=encoding_w.len()/numeric.len();
    if bins<2||weight0.len()!=numeric.len()*d||bias0.len()!=weight0.len()||encoding_b.len()!=encoding_w.len()
        ||embedding_w.len()!=numeric.len()*bins*d{return Err(invalid("Embedding tensor shape differs"))}
    let mut result=Vec::with_capacity(numeric.len()*d+bypass.len());
    for (feature,&index) in numeric.iter().enumerate(){
        let value=*x.get(index as usize).ok_or_else(||invalid("Invalid embedding input index"))?;
        let mut encoding=Vec::with_capacity(bins);
        for bin in 0..bins{
            // Keep both roundings. FMA changes narrow-bin encodings observably.
            let product=encoding_w[feature*bins+bin]*value;
            let linear=product+encoding_b[feature*bins+bin];
            encoding.push(if bin==0{linear.min(1.0)}else if bin+1==bins{linear.max(0.0)}else{linear.clamp(0.0,1.0)});
        }
        for dim in 0..d{
            let linear=value*weight0[feature*d+dim]+bias0[feature*d+dim];
            let piecewise=(0..bins).map(|bin|encoding[bin] as f64 * embedding_w[(feature*bins+bin)*d+dim] as f64).sum::<f64>() as f32;
            result.push(linear+piecewise);
        }
    }
    for &i in bypass{result.push(*x.get(i as usize).ok_or_else(||invalid("Invalid bypass index"))?)}
    Ok(result)
}

pub fn tabm(manifest:&Value,tensors:&Tensors,input:Vec<f32>)->Result<Vec<f64>,ModelError>{
    let cfg=&manifest["config"];let members=integer(&cfg["k"])?;let width=integer(&cfg["width"])?;
    let depth=integer(&cfg["depth"])?;let classes=manifest["classes"].as_array().ok_or_else(||invalid("Missing class schema"))?.len();
    if members==0||members>64||depth==0||depth>16||width==0||width>4096{return Err(invalid("TabM dimensions outside bounds"))}
    let embedded=embedding(manifest,tensors,&input)?;
    let hierarchical=manifest["kind"]!="tabm_ne";
    let mut probability=vec![0.0f64;classes];
    for member in 0..members{
        let mut x=embedded.clone();
        for layer in 0..depth{
            let prefix=format!("backbone.blocks.{layer}.0");
            let w=tensors.f32(&format!("{prefix}.weight"))?;let r=tensors.f32(&format!("{prefix}.r"))?;
            let s=tensors.f32(&format!("{prefix}.s"))?;let b=tensors.f32(&format!("{prefix}.bias"))?;
            let n=x.len();
            if w.len()!=width*n||r.len()!=members*n||s.len()!=members*width||b.len()!=s.len(){return Err(invalid("TabM block shape differs"))}
            for i in 0..n{x[i]*=r[member*n+i]}
            x=(0..width).map(|o|(dot(&x,&w[o*n..(o+1)*n])*s[member*width+o]+b[member*width+o]).max(0.0)).collect();
        }
        let w=tensors.f32("head.weight")?;let b=tensors.f32("head.bias")?;
        if w.len()!=members*width*classes||b.len()!=members*classes{return Err(invalid("TabM head shape differs"))}
        let logits:Vec<f32>=(0..classes).map(|c|{
            let value=(0..width).map(|i|x[i] as f64*w[(member*width+i)*classes+c] as f64).sum::<f64>() as f32;
            value+b[member*classes+c]
        }).collect();
        let p=if hierarchical{
            let attack=sigmoid(logits[0] as f64) as f32;let subtype=softmax(&logits[1..]);
            let mut values=vec![(1.0-attack) as f64];values.extend(subtype.iter().map(|p|(*p as f32*attack) as f64));values
        }else{softmax(&logits)};
        for c in 0..classes{probability[c]+=p[c]/members as f64}
    }
    Ok(probability.iter().map(|p|(*p as f32) as f64).collect())
}

pub fn tabpack(manifest:&Value,tensors:&Tensors,input:Vec<f32>)->Result<Vec<f64>,ModelError>{
    let cfg=&manifest["config"];let members=integer(&cfg["pack_size"])?;
    let widths:Vec<usize>=cfg["d_block"].as_array().ok_or_else(||invalid("Missing pack widths"))?.iter().map(integer).collect::<Result<_,_>>()?;
    let depths:Vec<usize>=cfg["n_blocks"].as_array().ok_or_else(||invalid("Missing pack depths"))?.iter().map(integer).collect::<Result<_,_>>()?;
    let weights:Vec<f64>=manifest["member_weights"].as_array().ok_or_else(||invalid("Missing pack weights"))?.iter()
        .map(|p|p.as_f64().ok_or_else(||invalid("Invalid pack weight"))).collect::<Result<_,_>>()?;
    let classes=manifest["classes"].as_array().ok_or_else(||invalid("Missing classes"))?.len();
    if widths.len()!=members||depths.len()!=members||weights.len()!=members||members>64{return Err(invalid("Pack dimensions differ"))}
    let maximum=*widths.iter().max().ok_or_else(||invalid("Empty pack"))?;
    let mut output=vec![0.0f64;classes];
    for m in 0..members{
        if weights[m]==0.0{continue}
        let width=widths[m];let mut x=input.clone();
        if width==0||depths[m]==0||depths[m]>16||maximum>4096{return Err(invalid("Pack dimensions outside bounds"))}
        for layer in 0..depths[m]{
            let p=format!("backbone.blocks.{layer}.linear");let w=tensors.f32(&format!("{p}.weight"))?;let b=tensors.f32(&format!("{p}.bias"))?;
            let max_input=if layer==0{input.len()}else{maximum};
            if w.len()!=members*maximum*max_input||b.len()!=members*maximum{return Err(invalid("Pack tensor shape differs"))}
            x=(0..width).map(|o|(dot(&x,&w[(m*maximum+o)*max_input..(m*maximum+o)*max_input+x.len()])+b[m*maximum+o]).max(0.0)).collect();
        }
        let w=tensors.f32("output.weight")?;let b=tensors.f32("output.bias")?;
        if w.len()!=members*classes*maximum||b.len()!=members*classes{return Err(invalid("Pack head shape differs"))}
        let logits:Vec<f32>=(0..classes).map(|o|dot(&x,&w[(m*classes+o)*maximum..(m*classes+o)*maximum+x.len()])+b[m*classes+o]).collect();
        let p=softmax(&logits);for c in 0..classes{output[c]+=weights[m]*p[c]}
    }
    Ok(output.iter().map(|p|(*p as f32) as f64).collect())
}
