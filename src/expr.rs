use crate::*;

pub trait Expression<S: RelationSymbol, T: Data> {
    fn arity(&self) -> usize;

    fn eval<F>(&self, ctx: &Database<S, T>, f: F)
    where
        F: FnMut(&[T]);

    fn eval_to_vecs(&self, ctx: &Database<S, T>) -> Vec<Vec<T>>
where {
        let mut v = vec![];
        self.eval(ctx, |tup| v.push(tup.to_vec()));
        v
    }

    fn into_dyn(self) -> DynExpression<S, T>
    where
        Self: Sized + 'static,
    {
        DynExpression(Box::new(self))
    }
}

pub struct DynExpression<S: RelationSymbol, T: Data>(Box<dyn hack::DynExpTrait<S, T>>);

mod hack {
    use super::*;
    pub trait DynExpTrait<S: RelationSymbol, T: Data> {
        fn dyn_arity(&self) -> usize;
        fn dyn_eval(&self, ctx: &Database<S, T>, f: &mut dyn FnMut(&[T]));
    }

    impl<S, T, E> DynExpTrait<S, T> for E
    where
        S: RelationSymbol,
        T: Data,
        E: Expression<S, T>,
    {
        fn dyn_arity(&self) -> usize {
            Expression::arity(self)
        }

        fn dyn_eval(&self, ctx: &Database<S, T>, f: &mut dyn FnMut(&[T])) {
            Expression::eval(self, ctx, f)
        }
    }

    impl<S: RelationSymbol, T: Data> Expression<S, T> for DynExpression<S, T> {
        fn arity(&self) -> usize {
            self.0.dyn_arity()
        }

        fn eval<F>(&self, ctx: &Database<S, T>, mut f: F)
        where
            F: FnMut(&[T]),
        {
            self.0.dyn_eval(ctx, &mut f)
        }
    }
}

#[derive(Debug, Clone)]
pub struct Scan<S> {
    pub relation: S,
    pub arity: usize,
}

impl<S> Scan<S> {
    pub fn new(relation: S, arity: usize) -> Self {
        Self { relation, arity }
    }
}

impl<S: RelationSymbol, T: Data> Expression<S, T> for Scan<S> {
    fn arity(&self) -> usize {
        self.arity
    }

    fn eval<F>(&self, ctx: &Database<S, T>, f: F)
    where
        Self: Sized,
        F: FnMut(&[T]),
    {
        let r = &ctx[&self.relation];
        assert_eq!(self.arity, r.arity);
        r.iter().for_each(f)
    }
}

#[derive(Debug, Clone)]
pub struct HashJoin<Small, Big> {
    pub small_key: usize,
    pub big_key: usize,
    pub small: Small,
    pub big: Big,
}

impl<S, T, Small, Big> Expression<S, T> for HashJoin<Small, Big>
where
    S: RelationSymbol,
    T: Data,
    Small: Expression<S, T>,
    Big: Expression<S, T>,
{
    fn arity(&self) -> usize {
        self.small.arity() + self.big.arity()
    }

    fn eval<F>(&self, ctx: &Database<S, T>, mut f: F)
    where
        Self: Sized,
        F: FnMut(&[T]),
    {
        let mut vec = vec![T::default(); self.arity()];
        let mut h: HashMap<T, Vec<T>> = Default::default();
        self.small.eval(ctx, |tup| {
            let k = tup[self.small_key].clone();
            h.entry(k).or_default().extend_from_slice(tup);
        });

        let sn = self.small.arity();
        self.big.eval(ctx, |tup| {
            if let Some(smalls) = h.get(&tup[self.big_key]) {
                for small in smalls.chunks(sn) {
                    vec[..sn].clone_from_slice(small);
                    vec[sn..].clone_from_slice(tup);
                    f(&vec);
                }
            }
        })
    }
}

pub struct EqFilter<Inner> {
    pub keys: (usize, usize),
    pub inner: Inner,
}

impl<S, T, Inner> Expression<S, T> for EqFilter<Inner>
where
    S: RelationSymbol,
    T: Data,
    Inner: Expression<S, T>,
{
    fn arity(&self) -> usize {
        self.inner.arity()
    }

    fn eval<F>(&self, ctx: &Database<S, T>, mut f: F)
    where
        Self: Sized,
        F: FnMut(&[T]),
    {
        self.inner.eval(ctx, |tup| {
            if tup[self.keys.0] == tup[self.keys.1] {
                f(tup)
            }
        })
    }
}

pub struct CheckConstant<T, Inner> {
    pub constant: T,
    pub key: usize,
    pub inner: Inner,
}

impl<S, T, Inner> Expression<S, T> for CheckConstant<T, Inner>
where
    S: RelationSymbol,
    T: Data,
    Inner: Expression<S, T>,
{
    fn arity(&self) -> usize {
        self.inner.arity()
    }

    fn eval<F>(&self, ctx: &Database<S, T>, mut f: F)
    where
        Self: Sized,
        F: FnMut(&[T]),
    {
        self.inner.eval(ctx, |tup| {
            if tup[self.key] == self.constant {
                f(tup)
            }
        })
    }
}
