use crate::*;
use fxhash::FxHashMap;
use std::sync::Arc;

pub trait Tuple<T: Data>: Into<Vec<T>> + AsRef<[T]> + Hash + Eq + Debug {}
impl<T: Data> Tuple<T> for Vec<T> {}
impl<T: Data> Tuple<T> for [T; 1] {}
impl<T: Data> Tuple<T> for [T; 2] {}
impl<T: Data> Tuple<T> for [T; 3] {}

pub trait Picker<T: Data>: Debug {
    type Out: Tuple<T>;
    fn pick(&self, t: &[T]) -> Self::Out;
}

impl<T: Data> Picker<T> for [usize; 1] {
    type Out = [T; 1];

    #[inline(always)]
    fn pick(&self, t: &[T]) -> Self::Out {
        [grab(t, self[0])]
    }
}

#[inline(always)]
fn grab<T: Clone>(ts: &[T], i: usize) -> T {
    // unsafe { ts.get_unchecked(i) }.clone()
    ts[i].clone()
}

impl<T: Data> Picker<T> for [usize; 2] {
    type Out = [T; 2];

    #[inline(always)]
    fn pick(&self, t: &[T]) -> Self::Out {
        [grab(t, self[0]), grab(t, self[1])]
    }
}

impl<T: Data> Picker<T> for Vec<usize> {
    type Out = Vec<T>;

    fn pick(&self, ts: &[T]) -> Self::Out {
        self.iter().copied().map(|i| grab(ts, i)).collect()
    }
}

pub trait Picker2<T: Data>: Debug {
    type Out: Tuple<T>;
    fn pick2(&self, a: &[T], b: &[T]) -> Self::Out;
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Sided<T> {
    Left(T),
    Right(T),
}

impl<T: Data> Picker2<T> for Vec<Sided<usize>> {
    type Out = Vec<T>;

    fn pick2(&self, a: &[T], b: &[T]) -> Self::Out {
        self.iter()
            .map(|i| match i {
                Sided::Left(i) => grab(a, *i),
                Sided::Right(i) => grab(b, *i),
            })
            .collect()
    }
}

pub trait Expression<DB: Database>: Debug {
    type Tuple: Tuple<DB::T>;

    fn eval<F>(&self, db: &DB, f: F)
    where
        F: FnMut(Self::Tuple);

    fn eval_ref<F>(&self, db: &DB, mut f: F)
    where
        F: FnMut(&[DB::T]),
    {
        self.eval(db, |x| f(x.as_ref()))
    }

    fn size_hint(&self, _db: &DB) -> (usize, usize);

    fn collect(&self, db: &DB) -> Vec<Self::Tuple> {
        let mut v = vec![];
        self.eval(db, |x| v.push(x));
        v
    }

    fn into_dyn(self) -> DynExpression<DB>
    where
        Self: Sized + 'static,
    {
        DynExpression(Arc::new(self))
    }
}

pub struct DynExpression<DB: Database>(Arc<dyn dynexpr::DynExpressionTrait<DB>>);

impl<DB: Database> Clone for DynExpression<DB> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<DB: Database> Debug for DynExpression<DB> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

mod dynexpr {
    use super::*;

    pub trait DynExpressionTrait<DB: Database>: Debug {
        fn dyn_eval(&self, db: &DB, f: &mut dyn FnMut(Vec<DB::T>));
        fn dyn_eval_ref(&self, db: &DB, f: &mut dyn FnMut(&[DB::T]));
        fn dyn_size_hint(&self, db: &DB) -> (usize, usize);
    }

    impl<DB, E> DynExpressionTrait<DB> for E
    where
        DB: Database,
        E: Expression<DB>,
    {
        fn dyn_eval(&self, db: &DB, f: &mut dyn FnMut(Vec<DB::T>)) {
            self.eval(db, |x| f(x.into()));
        }

        fn dyn_eval_ref(&self, db: &DB, f: &mut dyn FnMut(&[DB::T])) {
            self.eval_ref(db, f);
        }

        fn dyn_size_hint(&self, db: &DB) -> (usize, usize) {
            self.size_hint(db)
        }
    }

    impl<DB: Database> Expression<DB> for DynExpression<DB> {
        type Tuple = Vec<DB::T>;

        fn eval<F>(&self, db: &DB, mut f: F)
        where
            F: FnMut(Self::Tuple),
        {
            self.0.dyn_eval(db, &mut f);
        }

        fn eval_ref<F>(&self, db: &DB, mut f: F)
        where
            F: FnMut(&[DB::T]),
        {
            self.0.dyn_eval_ref(db, &mut f)
        }

        fn size_hint(&self, db: &DB) -> (usize, usize) {
            self.0.dyn_size_hint(db)
        }
    }
}

#[derive(Debug, Clone)]
pub struct Scan<S, T> {
    pub var_eqs: Vec<(usize, usize)>,
    pub term_eqs: Vec<(T, usize)>,
    pub relation: S,
}

impl<S, T> Scan<S, T> {
    pub fn new(relation: S) -> Self {
        Self {
            relation,
            var_eqs: vec![],
            term_eqs: vec![],
        }
    }
}

impl<DB: Database> Expression<DB> for Scan<DB::S, DB::T> {
    type Tuple = Vec<DB::T>;

    fn eval<F>(&self, db: &DB, mut f: F)
    where
        F: FnMut(Self::Tuple),
    {
        self.eval_ref(db, |x| f(x.to_owned()));
    }

    fn eval_ref<F>(&self, db: &DB, f: F)
    where
        F: FnMut(&[DB::T]),
    {
        db.get(&self.relation)
            .filter(|x| self.var_eqs.iter().all(|(i, j)| x[*i] == x[*j]))
            .filter(|x| self.term_eqs.iter().all(|(t, i)| &x[*i] == t))
            .for_each(f)
    }

    fn size_hint(&self, db: &DB) -> (usize, usize) {
        let len = db.get(&self.relation).len();
        if len == 0 {
            return (0, 0);
        }

        if self.var_eqs.is_empty() && self.term_eqs.is_empty() {
            return (len, len);
        }

        (0, len)
    }
}

#[derive(Debug, Clone)]
pub struct HashJoin<K1, K2, E1, E2, M> {
    pub key1: K1,
    pub key2: K2,
    pub expr1: E1,
    pub expr2: E2,
    pub merge: M,
}

impl<K, Out, K1, K2, E1, E2, M, DB> Expression<DB> for HashJoin<K1, K2, E1, E2, M>
where
    K: Hash + Eq,
    Out: Clone + Tuple<DB::T>,
    DB: Database,
    E1: Expression<DB>,
    E2: Expression<DB>,
    K1: Picker<DB::T, Out = K>,
    K2: Picker<DB::T, Out = K>,
    M: Picker2<DB::T, Out = Out>,
{
    type Tuple = Out;

    fn eval<F>(&self, db: &DB, mut f: F)
    where
        F: FnMut(Self::Tuple),
    {
        if self.size_hint(db).1 == 0 {
            return;
        }

        let mut map: FxHashMap<K, Vec<E1::Tuple>> = Default::default();
        self.expr1.eval(db, |t1| {
            let k = self.key1.pick(t1.as_ref());
            map.entry(k).or_default().push(t1);
        });

        self.expr2.eval_ref(db, |t2| {
            let k = self.key2.pick(t2);
            if let Some(t1s) = map.get(&k) {
                for t1 in t1s {
                    let t: Out = self.merge.pick2(t1.as_ref(), t2);
                    f(t)
                }
            }
        });
    }

    fn size_hint(&self, db: &DB) -> (usize, usize) {
        let (_lo1, hi1) = self.expr1.size_hint(db);
        let (_lo2, hi2) = self.expr2.size_hint(db);
        (0, hi1.saturating_mul(hi2))
    }
}

#[allow(clippy::many_single_char_names)]
#[cfg(test)]
mod tests {
    use crate::*;
    use Term::{Constant as C, Variable as V};

    type DB = SimpleDatabase<&'static str, u32>;

    #[test]
    fn triangle() {
        let mut db = DB::default();

        // R(0,1) S(1,2) T(2, 0)

        let mut r = vec![];
        let mut s = vec![];
        let mut t = vec![];

        let mut triangles = vec![vec![0, 10, 20], vec![3, 44, 16], vec![77, 6, 31]];

        triangles.sort();
        triangles.dedup();

        for tri in &triangles {
            let (a, b, c) = (tri[0], tri[1], tri[2]);
            r.push(vec![a, b]);
            s.push(vec![b, c]);
            t.push(vec![c, a]);
        }

        // add some junk
        let junk = if cfg!(debug_assertions) { 100 } else { 10000 };
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

        db.map.insert("r", (2, r.concat()));
        db.map.insert("s", (2, s.concat()));
        db.map.insert("t", (2, t.concat()));

        use Sided::*;

        let q1 = HashJoin {
            key1: [0, 1],
            key2: [0, 1],
            merge: vec![Right(0), Right(1), Right(2)],
            expr1: Scan::new("r"),
            expr2: HashJoin {
                key1: [1],
                key2: [0],
                merge: vec![Right(1), Left(0), Left(1)],
                expr1: Scan::new("s"),
                expr2: Scan::new("t"),
            },
        };

        let q2 = Query::new(vec![
            Atom::new("r", vec![V(0), V(1)]),
            Atom::new("s", vec![V(1), V(2)]),
            Atom::new("t", vec![V(2), V(0)]),
        ])
        .compile();

        let n = 300;
        let test = |q: DynExpression<DB>| {
            let (mut results, times): (Vec<_>, Vec<_>) = std::iter::repeat_with(|| {
                let start = std::time::Instant::now();
                (q.collect(&db), start.elapsed())
            })
            .take(n)
            .unzip();
            println!("min time: {:?}", times.iter().min().unwrap());
            let mut result = results.pop().unwrap();
            result.sort();
            result.dedup();
            result
        };

        println!("{:?}", Expression::<DB>::into_dyn(q1.clone()));

        let result1 = test(q1.into_dyn());
        let result2 = test(q2.into_dyn());

        assert_eq!(result1, triangles);
        assert_eq!(result2, triangles);
    }

    #[test]
    fn equality() {
        let mut db = DB::default();

        let mut r = vec![];

        let n = 10;
        for i in 0..n {
            for j in 0..n {
                r.push(vec![i, j]);
            }
        }

        db.map.insert("r", (2, r.concat()));

        let q: DynExpression<DB> = Query::new(vec![
            Atom::new("r", vec![C(7), V("a")]),
            Atom::new("r", vec![V("a"), V("a")]),
        ])
        .compile();

        let expected = vec![vec![7]; n as usize];
        let actual = q.collect(&db);

        assert_eq!(expected, actual);
    }
}
