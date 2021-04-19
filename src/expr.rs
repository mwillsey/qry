use std::rc::Rc;

use crate::*;

#[derive(Debug, Clone, Hash, PartialEq, Eq, Copy)]
pub struct Sided(bool, u32);

impl Sided {
    pub fn left(i: usize) -> Self {
        Self(false, i as _)
    }
    pub fn right(i: usize) -> Self {
        Self(true, i as _)
    }
    #[inline(always)]
    fn choose<T: Data>(self, left: &[T], right: &[T]) -> T {
        let side = if self.0 { right } else { left };
        side[self.1 as usize].clone()
    }
}

pub type Key = Vec<usize>;
pub type Key2 = Vec<Sided>;

type MapData1<T> = HashMap<T, Vec<T>>;
type MapData2<T> = HashMap<[T; 2], Vec<T>>;
type MapData3<T> = HashMap<[T; 3], Vec<T>>;
type MapDataN<T> = HashMap<Vec<T>, Vec<T>>;

#[derive(Debug, Clone)]
enum KeyedMapKind<T> {
    A0(Vec<T>),
    A1(MapData1<T>),
    A2(MapData2<T>),
    A3(MapData3<T>),
    Vec(MapDataN<T>),
}

#[derive(Debug, Clone)]
pub struct KeyedMap<T> {
    arity: usize,
    data: KeyedMapKind<T>,
}

fn pick<T: Clone>(key: &[usize], tuple: &[T]) -> Vec<T> {
    key.iter().map(|&i| tuple[i].clone()).collect()
}

#[inline(always)]
fn merge_pick<'a, T: Data>(
    merge: &'a [Sided],
    left: &'a [T],
    right: &'a [T],
) -> impl Iterator<Item = T> + 'a {
    merge.iter().map(move |side| side.choose(left, right))
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Keyed<S, T> {
    pub key: Key,
    pub expr: Box<Expr<S, T>>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Expr<S, T> {
    Scan {
        relation: (S, usize),
        var_eqs: Vec<(usize, usize)>,
        term_eqs: Vec<(T, usize)>,
    },
    Join {
        left: Keyed<S, T>,
        right: Keyed<S, T>,
        merge: Vec<Sided>,
    },
}

#[derive(Debug, Clone)]
pub struct HashEvalContext<S, T> {
    cache: HashMap<Keyed<S, T>, Rc<KeyedMap<T>>>,
}

impl<S, T> HashEvalContext<S, T> {
    pub fn clear(&mut self) {
        self.cache.clear()
    }
}

impl<S, T> Default for HashEvalContext<S, T> {
    fn default() -> Self {
        Self {
            cache: Default::default(),
        }
    }
}

fn do_join<T, F>(left: Rc<KeyedMap<T>>, right: Rc<KeyedMap<T>>, mut f: F)
where
    T: Data,
    F: FnMut(&[T], &[T]),
{
    macro_rules! cross {
        ($tups1:ident, $tups2:ident) => {{
            for tup1 in $tups1.chunks_exact(left.arity) {
                for tup2 in $tups2.chunks_exact(right.arity) {
                    f(tup1, tup2)
                }
            }
        }};
    }

    macro_rules! join {
        ($l:ident, $r:ident) => {{
            if $l.len() <= $r.len() {
                for (k, ll) in $l {
                    if let Some(rr) = $r.get(k) {
                        cross!(ll, rr)
                    }
                }
            } else {
                for (k, rr) in $r {
                    if let Some(ll) = $l.get(k) {
                        cross!(ll, rr)
                    }
                }
            }
        }};
    }

    match (&left.data, &right.data) {
        (KeyedMapKind::A0(l), KeyedMapKind::A0(r)) => cross!(l, r),
        (KeyedMapKind::A1(l), KeyedMapKind::A1(r)) => join!(l, r),
        (KeyedMapKind::A2(l), KeyedMapKind::A2(r)) => join!(l, r),
        (KeyedMapKind::A3(l), KeyedMapKind::A3(r)) => join!(l, r),
        (KeyedMapKind::Vec(l), KeyedMapKind::Vec(r)) => join!(l, r),
        _ => panic!("{:?}\n{:?}", left, right),
    }
}

impl<S: RelationSymbol, T: Data> Expr<S, T> {
    pub fn arity(&self) -> usize {
        match self {
            Expr::Scan { relation, .. } => relation.1,
            Expr::Join { merge, .. } => merge.len(),
        }
    }

    pub fn eval(
        &self,
        key: &[usize],
        db: &Database<S, T>,
        ctx: &mut HashEvalContext<S, T>,
    ) -> KeyedMap<T> {
        match self {
            Expr::Scan {
                relation,
                var_eqs,
                term_eqs,
            } => {
                let iter = db.get(relation).filter(|tup| {
                    var_eqs.iter().all(|(i, j)| tup[*i] == tup[*j])
                        && term_eqs.iter().all(|(t, i)| t == &tup[*i])
                });
                let data = match key {
                    &[] => {
                        let mut data = vec![];
                        iter.for_each(|tup| data.extend_from_slice(tup));
                        KeyedMapKind::A0(data)
                    }
                    &[k1] => {
                        let mut data = MapData1::default();
                        iter.for_each(|tup| {
                            data.entry(tup[k1].clone())
                                .or_default()
                                .extend_from_slice(tup)
                        });
                        KeyedMapKind::A1(data)
                    }
                    &[k1, k2] => {
                        let mut data = MapData2::default();
                        iter.for_each(|tup| {
                            let k = [tup[k1].clone(), tup[k2].clone()];
                            data.entry(k).or_default().extend_from_slice(tup)
                        });
                        KeyedMapKind::A2(data)
                    }
                    &[k1, k2, k3] => {
                        let mut data = MapData3::default();
                        iter.for_each(|tup| {
                            let k = [tup[k1].clone(), tup[k2].clone(), tup[k3].clone()];
                            data.entry(k).or_default().extend_from_slice(tup)
                        });
                        KeyedMapKind::A3(data)
                    }
                    key => {
                        let mut data = MapDataN::default();
                        iter.for_each(|tup| {
                            data.entry(pick(key, tup))
                                .or_default()
                                .extend_from_slice(tup)
                        });
                        KeyedMapKind::Vec(data)
                    }
                };
                KeyedMap {
                    arity: relation.1,
                    data,
                }
            }
            Expr::Join { left, right, merge } => {
                assert_eq!(left.key.len(), right.key.len());

                let left = if let Some(hit) = ctx.cache.get(left) {
                    // println!("hit {:?}", left);
                    Rc::clone(hit)
                } else {
                    let out = Rc::new(left.expr.eval(&left.key, db, ctx));
                    ctx.cache.insert(left.clone(), out.clone());
                    out
                };
                let right = if let Some(hit) = ctx.cache.get(right) {
                    // println!("hit {:?}", right);
                    Rc::clone(hit)
                } else {
                    let out = Rc::new(right.expr.eval(&right.key, db, ctx));
                    ctx.cache.insert(right.clone(), out.clone());
                    out
                };
                let key_merge = pick(key, merge);
                let data = match key_merge.as_slice() {
                    [] => {
                        let mut data = vec![];
                        do_join(left, right, |l, r| data.extend(merge_pick(merge, l, r)));
                        KeyedMapKind::A0(data)
                    }
                    [k1] => {
                        let mut data = MapData1::default();
                        do_join(left, right, |l, r| {
                            data.entry(k1.choose(l, r))
                                .or_default()
                                .extend(merge_pick(merge, l, r))
                        });
                        KeyedMapKind::A1(data)
                    }
                    [k1, k2] => {
                        let mut data = MapData2::default();
                        do_join(left, right, |l, r| {
                            let k = [k1.choose(l, r), k2.choose(l, r)];
                            data.entry(k).or_default().extend(merge_pick(merge, l, r))
                        });
                        KeyedMapKind::A2(data)
                    }
                    [k1, k2, k3] => {
                        let mut data = MapData3::default();
                        do_join(left, right, |l, r| {
                            let k = [k1.choose(l, r), k2.choose(l, r), k3.choose(l, r)];
                            data.entry(k).or_default().extend(merge_pick(merge, l, r))
                        });
                        KeyedMapKind::A3(data)
                    }
                    _ => {
                        // panic!("Can't support keys of length {}, {:?}", key_merge.len(), key)
                        let mut data = MapDataN::default();
                        do_join(left, right, |l, r| {
                            let k = merge_pick(&key_merge, l, r).collect();
                            data.entry(k).or_default().extend(merge_pick(merge, l, r))
                        });
                        KeyedMapKind::Vec(data)
                    }
                };
                KeyedMap {
                    arity: merge.len(),
                    data,
                }
            }
        }
    }

    pub fn for_each<F>(&self, db: &Database<S, T>, ctx: &mut HashEvalContext<S, T>, f: F)
    where
        F: FnMut(&[T]),
    {
        let key = &[];
        let map = self.eval(key, db, ctx);
        match map.data {
            KeyedMapKind::A0(data) => data.chunks_exact(self.arity()).for_each(f),
            _ => unreachable!(),
        }
    }

    pub fn collect(
        &self,
        db: &Database<S, T>,
        ctx: &mut HashEvalContext<S, T>,
        picker: &[usize],
    ) -> Vec<Vec<T>> {
        let mut v = vec![];
        self.for_each(db, ctx, |tup| {
            v.push(picker.iter().map(|i| tup[*i].clone()).collect())
        });
        v
    }

    pub fn new_scan(symbol: S, arity: usize) -> Self {
        Self::Scan {
            relation: (symbol, arity),
            var_eqs: vec![],
            term_eqs: vec![],
        }
    }
}

#[allow(clippy::many_single_char_names)]
#[cfg(test)]
mod tests {
    use crate::*;
    use Term::Variable as V;

    // #[test]
    // fn simple() {
    //     let mut db = Database::<&'static str, i32>::default();

    //     db.add_relation_with_data("r", 2, vec![20, 30, 30, 40, 10, 20]);

    //     let (varmap, q) = Query::<_, _, i32>::new(vec![
    //         Atom::new("r", vec![V(0), V(1)]),
    //         Atom::new("r", vec![V(1), V(2)]),
    //         Atom::new("r", vec![V(2), V(3)]),
    //     ])
    //     .compile();

    //     let picker = &[varmap[&0], varmap[&1], varmap[&2], varmap[&3]];
    //     let result = q.collect(&db, &mut HashEvalContext::default(), picker);
    //     assert_eq!(result, vec![vec![10, 20, 30, 40]]);
    // }

    // #[test]
    // fn triangle() {
    //     let mut db = Database::<&'static str, i32>::default();

    //     // R(0,1) S(1,2) T(2, 0)

    //     let mut r = vec![];
    //     let mut s = vec![];
    //     let mut t = vec![];

    //     let mut triangles = vec![vec![100, 200, 300]];

    //     triangles.sort();
    //     triangles.dedup();

    //     for tri in &triangles {
    //         let (a, b, c) = (tri[0], tri[1], tri[2]);
    //         r.push(vec![a, b]);
    //         s.push(vec![b, c]);
    //         t.push(vec![c, a]);
    //     }

    //     // add some junk
    //     let junk = if cfg!(debug_assertions) { 10 } else { 10000 };
    //     for i in 0..junk {
    //         let j = i + 1;
    //         r.push(vec![i, j]);
    //         s.push(vec![i, j]);
    //         t.push(vec![i, j]);
    //     }

    //     r.sort();
    //     r.dedup();
    //     s.sort();
    //     s.dedup();
    //     t.sort();
    //     t.dedup();

    //     db.add_relation_with_data("r", 2, r.concat());
    //     db.add_relation_with_data("s", 2, s.concat());
    //     db.add_relation_with_data("t", 2, t.concat());

    //     let q2 = Query::<_, _, i32>::new(vec![
    //         Atom::new("r", vec![V(0), V(1)]),
    //         Atom::new("s", vec![V(1), V(2)]),
    //         Atom::new("t", vec![V(2), V(0)]),
    //     ]);

    //     let mut results = vec![];
    //     let varmap = vec![(0, 0), (1, 1), (2, 2)].into_iter().collect();
    //     let mut ctx = EvalContext::default();
    //     q2.join(&varmap, &db, &mut ctx, |x| results.push(x.to_vec()));

    //     assert_eq!(results, triangles);
    // }
}
