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
            .copied()
            .map(|i| match i {
                Sided::Left(i) => grab(a, i),
                Sided::Right(i) => grab(b, i),
            })
            .collect()
    }
}

pub trait Expression<DB: Database>: Debug {
    type Tuple: Tuple<DB::T>;
    // type Tuple;
    // fn arity(&self) -> usize;

    fn eval<F>(&self, db: &DB, f: F)
    where
        F: FnMut(Self::Tuple);

    fn eval_ref<F>(&self, db: &DB, mut f: F)
    where
        F: FnMut(&[DB::T]),
    {
        self.eval(db, |x| f(x.as_ref()))
    }

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

#[derive(Clone)]
pub struct DynExpression<DB: Database>(Arc<dyn dynexpr::DynExpressionTrait<DB>>);

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
    }
}

#[derive(Debug, Clone)]
pub struct Scan<S> {
    relation: S,
}

impl<S> Scan<S> {
    pub fn new(relation: S) -> Self {
        Self { relation }
    }
}

impl<DB: Database> Expression<DB> for Scan<DB::S> {
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
        db.get(&self.relation).for_each(f)
    }
}

pub struct Filter<P, E> {
    pred: P,
    expr: E,
}

use std::fmt;

impl<P, E: Debug> fmt::Debug for Filter<P, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Filter").field("expr", &self.expr).finish()
    }
}

impl<P, E, DB: Database> Expression<DB> for Filter<P, E>
where
    E: Expression<DB>,
    P: Debug + Fn(&E::Tuple) -> bool,
{
    type Tuple = E::Tuple;

    fn eval<F>(&self, db: &DB, mut f: F)
    where
        F: FnMut(Self::Tuple),
    {
        self.expr.eval(db, |t| {
            if (self.pred)(&t) {
                f(t)
            }
        });
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
}

#[allow(clippy::many_single_char_names)]
#[cfg(test)]
mod tests {
    use super::*;
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

        let atom = |s: &'static str, vars: Vec<Var>| Atom {
            symbol: s,
            terms: vars.into_iter().map(Term::Variable).collect(),
        };

        let q2 = Query::new(vec![
            atom("r", vec![0, 1]),
            atom("s", vec![1, 2]),
            atom("t", vec![2, 0]),
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
}
