use std::rc::Rc;

use crate::*;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Sided<T> {
    Left(T),
    Right(T),
}

pub type Key = Vec<usize>;
pub type Key2 = Vec<Sided<usize>>;
pub type KeyedMap<T> = FxHashMap<Vec<T>, Vec<Vec<T>>>;

fn pick<T: Clone>(key: &[usize], tuple: &[T]) -> Vec<T> {
    key.iter().map(|&i| tuple[i].clone()).collect()
}

fn pick2<T: Clone>(merge: &[Sided<usize>], left: &[T], right: &[T]) -> Vec<T> {
    merge
        .iter()
        .map(|side| match side {
            Sided::Left(i) => left[*i].clone(),
            Sided::Right(i) => right[*i].clone(),
        })
        .collect()
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Keyed<S, T> {
    pub key: Key,
    pub expr: Box<Expr<S, T>>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Expr<S, T> {
    Scan {
        relation: S,
        var_eqs: Vec<(usize, usize)>,
        term_eqs: Vec<(T, usize)>,
    },
    Join {
        left: Keyed<S, T>,
        right: Keyed<S, T>,
        merge: Vec<Sided<usize>>,
    },
}

#[derive(Debug, Clone)]
pub struct EvalContext<S, T> {
    cache: FxHashMap<Keyed<S, T>, Rc<KeyedMap<T>>>,
}

impl<S, T> EvalContext<S, T> {
    pub fn clear(&mut self) {
        self.cache.clear()
    }
}

impl<S, T> Default for EvalContext<S, T> {
    fn default() -> Self {
        Self {
            cache: Default::default(),
        }
    }
}

impl<S: RelationSymbol, T: Data> Expr<S, T> {
    pub fn eval(
        &self,
        key: &[usize],
        db: &Database<S, T>,
        ctx: &mut EvalContext<S, T>,
    ) -> KeyedMap<T> {
        let mut map = KeyedMap::default();
        match self {
            Expr::Scan {
                relation,
                var_eqs,
                term_eqs,
            } => {
                for tup in db.get(relation) {
                    if var_eqs.iter().all(|(i, j)| tup[*i] == tup[*j])
                        && term_eqs.iter().all(|(t, i)| t == &tup[*i])
                    {
                        map.entry(pick(key, tup)).or_default().push(tup.to_vec())
                    }
                }
            }
            Expr::Join { left, right, merge } => {
                assert_eq!(left.key.len(), right.key.len());
                let left = if let Some(left) = ctx.cache.get(left) {
                    left.clone()
                } else {
                    let out = Rc::new(left.expr.eval(&left.key, db, ctx));
                    ctx.cache.insert(left.clone(), out.clone());
                    out
                };
                let right = if let Some(right) = ctx.cache.get(right) {
                    right.clone()
                } else {
                    let out = Rc::new(right.expr.eval(&right.key, db, ctx));
                    ctx.cache.insert(right.clone(), out.clone());
                    out
                };
                for (k1, vals1) in left.iter() {
                    if let Some(vals2) = right.get(k1) {
                        for val1 in vals1 {
                            for val2 in vals2 {
                                let merged = pick2(merge, val1, val2);
                                map.entry(pick(key, &merged)).or_default().push(merged);
                            }
                        }
                    }
                }
            }
        }
        debug_assert!(map.is_empty() || map.keys().next().unwrap().len() == key.len());
        map
    }

    pub fn collect(&self, db: &Database<S, T>, ctx: &mut EvalContext<S, T>) -> Vec<Vec<T>> {
        let key = &[];
        let mut map = self.eval(key, db, ctx);
        map.remove(&vec![]).unwrap_or_default()
    }

    pub fn new_scan(relation: S) -> Self {
        Self::Scan {
            relation,
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

    #[test]
    fn simple() {
        let mut db = Database::<&'static str, i32>::default();

        db.add_relation_with_data("r", 2, vec![2, 3, 3, 4, 1, 2]);

        let q = Query::<_, _, i32>::new(vec![
            Atom::new("r", vec![V(0), V(1)]),
            Atom::new("r", vec![V(1), V(2)]),
            Atom::new("r", vec![V(2), V(3)]),
        ])
        .compile();

        let result = q.collect(&db, &mut EvalContext::default());
        assert_eq!(result, vec![vec![1, 2, 3, 4]]);
    }

    #[test]
    fn triangle() {
        let mut db = Database::<&'static str, i32>::default();

        // R(0,1) S(1,2) T(2, 0)

        let mut r = vec![];
        let mut s = vec![];
        let mut t = vec![];

        let mut triangles = vec![vec![100, 200, 300]];

        triangles.sort();
        triangles.dedup();

        for tri in &triangles {
            let (a, b, c) = (tri[0], tri[1], tri[2]);
            r.push(vec![a, b]);
            s.push(vec![b, c]);
            t.push(vec![c, a]);
        }

        // add some junk
        let junk = if cfg!(debug_assertions) { 10 } else { 10000 };
        for i in 0..junk {
            let j = i + 1;
            r.push(vec![i, j]);
            s.push(vec![i, j]);
            t.push(vec![i, j]);
        }

        r.sort();
        r.dedup();
        s.sort();
        s.dedup();
        t.sort();
        t.dedup();

        db.add_relation_with_data("r", 2, r.concat());
        db.add_relation_with_data("s", 2, s.concat());
        db.add_relation_with_data("t", 2, t.concat());

        use Sided::*;

        let q1 = Expr::Join {
            merge: vec![Right(0), Right(1), Right(2)],
            left: Keyed {
                key: vec![0, 1],
                expr: Box::new(Expr::new_scan("r")),
            },
            right: Keyed {
                key: vec![0, 1],
                expr: Box::new(Expr::Join {
                    merge: vec![Right(1), Left(0), Left(1)],
                    left: Keyed {
                        key: vec![1],
                        expr: Box::new(Expr::new_scan("s")),
                    },
                    right: Keyed {
                        key: vec![0],
                        expr: Box::new(Expr::new_scan("t")),
                    },
                }),
            },
        };

        let q2 = Query::<_, _, i32>::new(vec![
            Atom::new("r", vec![V(0), V(1)]),
            Atom::new("s", vec![V(1), V(2)]),
            Atom::new("t", vec![V(2), V(0)]),
        ])
        .compile();

        let n = 300;
        let test = |q: Expr<_, _>| {
            let (mut results, times): (Vec<_>, Vec<_>) = std::iter::repeat_with(|| {
                let start = std::time::Instant::now();
                (q.collect(&db, &mut EvalContext::default()), start.elapsed())
            })
            .take(n)
            .unzip();
            println!("min time: {:?}", times.iter().min().unwrap());
            let mut result = results.pop().unwrap();
            result.sort();
            result.dedup();
            result
        };

        // println!("{:?}", Expression::<DB>::into_dyn(q1.clone()));

        let result1 = test(q1);
        let result2 = test(q2);

        assert_eq!(result1, triangles);
        assert_eq!(result2, triangles);
    }

    // #[test]
    // fn equality() {
    //     let mut db = DB::default();

    //     let mut r = vec![];

    //     let n = 10;
    //     for i in 0..n {
    //         for j in 0..n {
    //             r.push(vec![i, j]);
    //         }
    //     }

    //     db.map.insert("r", (2, r.concat()));

    //     let q: DynExpression<DB> = Query::new(vec![
    //         Atom::new("r", vec![C(7), V("a")]),
    //         Atom::new("r", vec![V("a"), V("a")]),
    //     ])
    //     .compile();

    //     let expected = vec![vec![7]; n as usize];
    //     let actual = q.collect(&db);

    //     assert_eq!(expected, actual);
    // }
}
