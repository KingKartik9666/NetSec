//! Small deterministic numerical helpers; no global RNG or linear-algebra runtime.

pub(crate) struct Rng(u64);
impl Rng {
    pub(crate) fn new(seed:u64)->Self{Self(seed)}
    fn next(&mut self)->u64{
        self.0=self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z=self.0;z=(z^(z>>30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z=(z^(z>>27)).wrapping_mul(0x94d049bb133111eb);z^(z>>31)
    }
    pub(crate) fn index(&mut self,n:usize)->usize{(self.next() % n as u64) as usize}
    pub(crate) fn coin(&mut self)->bool{self.next() & 1==1}
    pub(crate) fn shuffle<T>(&mut self,values:&mut [T]){
        for i in (1..values.len()).rev(){values.swap(i,self.index(i+1));}
    }
}

pub(crate) fn standard_error(values:&[f64])->Option<f64>{
    if values.len()<2{return None;}
    let n=values.len() as f64;let mean=values.iter().sum::<f64>()/n;
    Some((values.iter().map(|v|(v-mean).powi(2)).sum::<f64>()/(n*(n-1.))).sqrt())
}

pub(crate) fn wilson(successes:usize,n:usize)->(f64,f64){
    if n==0{return (0.,1.);}
    let n=n as f64;let p=successes as f64/n;let z=1.959963984540054;let z2=z*z;
    let center=(p+z2/(2.*n))/(1.+z2/n);
    let half=z*((p*(1.-p)/n+z2/(4.*n*n)).sqrt())/(1.+z2/n);
    ((center-half).max(0.),(center+half).min(1.))
}

/// Weighted least squares with a tiny ridge on coefficients, not the intercept.
pub(crate) fn ridge(x:&[Vec<f64>],y:&[f64],weights:&[f64])->Result<Vec<f64>,String>{
    let p=x[0].len()+1;let mut a=vec![vec![0.;p+1];p];
    for ((row,&target),&weight) in x.iter().zip(y).zip(weights){
        for i in 0..p {let vi=if i==0 {1.}else{row[i-1]};
            for j in 0..p {let vj=if j==0 {1.}else{row[j-1]};a[i][j]+=weight*vi*vj;}
            a[i][p]+=weight*vi*target;
        }
    }
    for (i,row) in a.iter_mut().enumerate().skip(1){row[i]+=1e-6;}
    for c in 0..p{
        let pivot=(c..p).max_by(|&i,&j|a[i][c].abs().total_cmp(&a[j][c].abs())).ok_or("Empty regression")?;
        a.swap(c,pivot);let d=a[c][c];
        if !d.is_finite()||d.abs()<1e-14{return Err("Local surrogate design is singular".into());}
        for j in c..=p {a[c][j]/=d;}
        for i in 0..p {if i!=c {let f=a[i][c];for j in c..=p{a[i][j]-=f*a[c][j];}}}
    }
    let solution=a.iter().map(|r|r[p]).collect::<Vec<_>>();
    if solution.iter().any(|v|!v.is_finite()){return Err("Nonfinite local regression coefficients".into());}
    Ok(solution)
}
