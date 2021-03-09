use std::{fmt::Debug, hash::Hash, rc::Rc, slice::ChunksExact};

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

mod expr;
pub use expr::*;

pub trait RelationSymbol: Debug + Clone + Hash + Eq {}
impl<T> RelationSymbol for T where T: Debug + Clone + Hash + Eq {}

pub trait Data: Debug + Clone + Hash + Eq {}
impl<T> Data for T where T: Debug + Clone + Hash + Eq {}

#[derive(Debug, Clone)]
pub struct Database<S, T> {
    pub relations: HashMap<(S, usize), Vec<T>>,
}

impl<S, T> Default for Database<S, T> {
    fn default() -> Self {
        let relations = Default::default();
        Self { relations }
    }
}

impl<S: RelationSymbol, T: Data> Database<S, T> {
    pub fn add_relation_with_data(&mut self, sym: S, arity: usize, data: Vec<T>) {
        self.relations.insert((sym, arity), data);
    }

    pub fn get(&self, key: &(S, usize)) -> ChunksExact<T> {
        if let Some(data) = self.relations.get(key) {
            data.chunks_exact(key.1)
        } else {
            (&[]).chunks_exact(1)
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Term<V, T> {
    Variable(V),
    Constant(T),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Atom<V, S, T> {
    pub symbol: S,
    pub arity: usize,
    pub terms: Vec<Term<V, T>>,
}

impl<V, S, T> Atom<V, S, T> {
    pub fn new(symbol: S, terms: Vec<Term<V, T>>) -> Self {
        Self {
            symbol,
            arity: terms.len(),
            terms,
        }
    }
}

impl<V: Eq + Clone, S, T> Atom<V, S, T> {
    pub fn vars(&self) -> impl Iterator<Item = V> + '_ {
        self.terms.iter().filter_map(|t| match t {
            Term::Variable(v) => Some(v.clone()),
            _ => None,
        })
    }

    pub fn has_var(&self, v: &V) -> bool {
        self.vars().position(|x| &x == v).is_some()
    }
}

pub type VarMap<V> = HashMap<V, usize>;

#[derive(Debug, Clone)]
pub struct Query<V, S, T> {
    pub atoms: Vec<Atom<V, S, T>>,
    by_var: HashMap<V, Vec<usize>>,
}

impl<V: PartialEq, S: PartialEq, T: PartialEq> PartialEq for Query<V, S, T> {
    fn eq(&self, other: &Self) -> bool {
        self.atoms == other.atoms
    }
}

impl<V, S, T> Query<V, S, T>
where
    V: Eq + Hash + Clone,
    S: RelationSymbol,
    T: Data,
{
    pub fn new(atoms: Vec<Atom<V, S, T>>) -> Self {
        let mut new = Self {
            atoms,
            by_var: Default::default(),
        };

        for (i, atom) in new.atoms.iter().enumerate() {
            for v in atom.vars() {
                let is = new.by_var.entry(v).or_default();
                if is.last() != Some(&i) {
                    is.push(i)
                }
            }
        }

        new
    }
}

impl<V, S, T> Query<V, S, T>
where
    V: Hash + Eq + Clone,
    S: RelationSymbol + 'static,
    T: Data + 'static,
{
    pub fn compile(&self) -> (VarMap<V>, Expr<S, T>) {
        let na = self.atoms.len();
        assert_ne!(na, 0);

        let mut exprs: Vec<(usize, usize, VarMap<V>, Expr<S, T>)> = self
            .atoms
            .iter()
            .map(|atom| {
                let mut used_vars: HashMap<V, Vec<usize>> = Default::default();
                let mut term_eqs: Vec<(T, usize)> = vec![];
                for (i, term) in atom.terms.iter().enumerate() {
                    match term {
                        Term::Variable(v) => used_vars.entry(v.clone()).or_default().push(i),
                        Term::Constant(c) => term_eqs.push((c.clone(), i)),
                    }
                }

                let mut var_map = VarMap::default();
                let mut var_eqs: Vec<(usize, usize)> = vec![];
                for (v, occurences) in used_vars {
                    var_map.insert(v, occurences[0]);
                    for pair in occurences.windows(2) {
                        var_eqs.push((pair[0], pair[1]))
                    }
                }

                let n_filters = var_eqs.len() + term_eqs.len();
                let expr = Expr::Scan {
                    relation: (atom.symbol.clone(), atom.arity),
                    var_eqs,
                    term_eqs,
                };
                (1, n_filters, var_map, expr)
            })
            .collect();

        while exprs.len() > 1 {
            let mut proposals = vec![];
            let mut iter = exprs.iter().enumerate();
            while let Some((i1, (d1, nf1, v1, _e1))) = iter.next() {
                for (i2, (d2, nf2, v2, _e2)) in iter.clone() {
                    let nf = nf1 + nf2;
                    let mut score = 0;
                    let mut var_map = VarMap::default();
                    let mut merge: Vec<Sided> = vec![];
                    for (v, i) in v1.iter() {
                        var_map.insert(v.clone(), merge.len());
                        merge.push(Sided::left(*i));
                    }
                    for (v, i) in v2.iter() {
                        if !v1.contains_key(v) {
                            var_map.insert(v.clone(), merge.len());
                            merge.push(Sided::right(*i));
                        } else {
                            score += 1;
                        }
                    }
                    proposals.push(((d1.max(d2) + 1, score, nf), var_map, merge, (i1, i2)));
                }
            }
            let ((depth, score, nf), var_map, merge, (i1, i2)) =
                proposals.iter().max_by_key(|p| p.0).unwrap().clone();
            dbg!(score, nf);
            assert!(i1 < i2);
            // must remove i2 first to not mess up index
            let (_, nf1, v2, e2) = exprs.remove(i2);
            let (_, nf2, v1, e1) = exprs.remove(i1);
            let (key1, key2): (Vec<usize>, Vec<usize>) = v1
                .iter()
                .filter_map(|(v, i1)| v2.get(v).map(|i2| (i1, i2)))
                .unzip();
            assert_eq!(score, key1.len());
            let expr = Expr::Join {
                left: Keyed {
                    key: key1,
                    expr: Box::new(e1),
                },
                right: Keyed {
                    key: key2,
                    expr: Box::new(e2),
                },
                merge,
            };
            exprs.push((depth, nf, var_map, expr));
        }

        println!("Compiled");

        assert_eq!(exprs.len(), 1);
        let (_, _nf, varmap, expr) = exprs.pop().unwrap();
        (varmap, expr)
    }
}

#[derive(Debug, Clone)]
struct Trie<T>(HashMap<T, Self>);

impl<T> Default for Trie<T> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<T> Trie<T> {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<T: Data> Trie<T> {
    fn insert(&mut self, shuffle: &[usize], tuple: &[T]) {
        debug_assert_eq!(shuffle.len(), tuple.len());
        let mut trie = self;
        for i in shuffle {
            trie = trie.0.entry(tuple[*i].clone()).or_default();
        }
    }
}

    // pub fn for_each<F>(&self, db: &Database<S, T>, ctx: &mut EvalContext<S, T>, f: F)
    // where
    //     F: FnMut(&[T]),
    // {
    //     let key = &[];
    //     let map = self.eval(key, db, ctx);
    //     match map.data {
    //         KeyedMapKind::A0(data) => data.chunks_exact(self.arity()).for_each(f),
    //         _ => unreachable!(),
    //     }
    // }


#[derive(Debug, Clone)]
pub struct EvalContext<S, T> {
    cache: HashMap<(S, usize, Vec<usize>), Rc<Trie<T>>>,
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

impl<V, S, T> Query<V, S, T>
where
    V: Eq + Hash + Clone + Debug,
    S: RelationSymbol,
    T: Data,
{
    pub fn vars(&self, _db: &Database<S, T>) -> VarMap<V> {
        let mut vars: Vec<_> = self
            .by_var
            .iter()
            .collect();

        vars.sort_by_key(|(_v, occ)| occ.len());

        vars.into_iter()
            .rev()
            .enumerate()
            .map(|(i, (v, _occ))| (v.clone(), i))
            .collect()
    }

    pub fn join<F>(&self, varmap: &VarMap<V>, db: &Database<S, T>, ctx: &mut EvalContext<S, T>, mut f: F)
    where
        F: FnMut(&[T]),
    {
        let mut vars: Vec<_> = self
            .by_var
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        vars.sort_by_key(|x| varmap[&x.0]);

        let tries = self.atoms.iter().map(|atom| {
            let mut shuffle = vec![];
            for (var, _) in &vars {
                for (i, term) in atom.terms.iter().enumerate() {
                    if let Term::Variable(v) = term {
                        if var == v && !shuffle.contains(&i) {
                            shuffle.push(i);
                        }
                    }
                }
            }

            let key = (atom.symbol.clone(), atom.arity, shuffle.clone());
            let trie = ctx.cache.entry(key).or_insert_with_key(|(_sym, _arity_, shuffle)| {
                let mut trie = Trie::default();
                for tuple in db.get(&(atom.symbol.clone(), atom.arity)) {
                    trie.insert(&shuffle, tuple);
                }
                Rc::new(trie)
            });

            // println!("{:?} {:?}", atom.symbol, &shuffle);
            trie.clone()
        }).collect::<Vec<_>>();

        let tries: Vec<&Trie<T>> = tries.iter().map(|t| t.as_ref()).collect();

        self.gj(&mut f, &[], &vars, &tries);
    }

    fn gj<F>(&self, f: &mut F, tuple: &[T], vars: &[(V, Vec<usize>)], relations: &[&Trie<T>])
    where
        F: FnMut(&[T]),
    {
        // println!("{:?}", tuple);
        if tuple.len() == vars.len() {
            return f(tuple);
        }

        assert!(tuple.len() < vars.len());

        let (x, js) = &vars[tuple.len()];
        debug_assert!(js.iter().all(|&j| self.atoms[j].has_var(x)));

        let j_min = js
            .iter()
            .copied()
            .min_by_key(|j| relations[*j].len())
            .unwrap();

        // for &j in js {
        //     println!("{:?}", relations[j].0.keys());
        // }

        let mut intersection: Vec<T> = relations[j_min].0.keys().cloned().collect();

        for &j in js {
            if j != j_min {
                let rj = &relations[j].0;
                intersection.retain(|t| rj.contains_key(t));
            }
        }

        // println!("intersection of {:?}: {:?}", x, intersection);

        let empty = Trie::default();

        let mut tuple = tuple.to_vec();
        for val in intersection {
            let relations: Vec<_> = relations
                .iter()
                .zip(&self.atoms)
                .map(|(r, a)| {
                    if a.has_var(x) {
                        r.0.get(&val).unwrap_or(&empty)
                    } else {
                        r
                    }
                })
                .collect();
            tuple.push(val);
            self.gj(f, &tuple, vars, &relations);
            tuple.pop();
        }
    }
}
