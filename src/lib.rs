use fxhash::{FxHashMap, FxHashSet};
use std::{fmt::Debug, hash::Hash, rc::Rc, slice::ChunksExact};

mod expr;
pub use expr::*;

pub trait RelationSymbol: Debug + Clone + Hash + Eq {}
impl<T> RelationSymbol for T where T: Debug + Clone + Hash + Eq {}

pub trait Data: Debug + Clone + Hash + Eq {}
impl<T> Data for T where T: Debug + Clone + Hash + Eq {}

#[derive(Debug, Clone)]
pub struct Database<S, T> {
    pub relations: FxHashMap<(S, usize), Vec<T>>,
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

impl<V: Clone, S, T> Atom<V, S, T> {
    pub fn new(symbol: S, terms: Vec<Term<V, T>>) -> Self {
        Self {
            symbol,
            arity: terms.len(),
            terms,
        }
    }

    pub fn vars(&self) -> impl Iterator<Item = V> + '_ {
        self.terms.iter().filter_map(|t| match t {
            Term::Variable(v) => Some(v.clone()),
            _ => None,
        })
    }
}

pub type VarMap<V> = FxHashMap<V, usize>;

#[derive(Debug, Clone, PartialEq)]
pub struct Query<V, S, T> {
    pub atoms: Vec<Atom<V, S, T>>,
}

impl<V, S, T> Query<V, S, T>
where
    V: Eq + Hash + Clone,
    S: RelationSymbol,
    T: Data,
{
    pub fn new(atoms: Vec<Atom<V, S, T>>) -> Self {
        Self { atoms }
    }
}

impl<V, S, T> Query<V, S, T>
where
    V: Hash + Eq + Clone,
    S: RelationSymbol + 'static,
    T: Data + 'static,
{
    pub fn compile(&self) -> (VarMap<V>, Expr<S, T>) {
        assert!(!self.atoms.is_empty());

        let mut by_var: FxHashMap<V, FxHashSet<usize>> = Default::default();
        for (i, atom) in self.atoms.iter().enumerate() {
            for var in atom.vars() {
                by_var.entry(var.clone()).or_default().insert(i);
            }
        }

        // Use Kruskal's algorithm to find a max spanning tree
        let mut edges = vec![];
        let mut nodes = self.atoms.iter().enumerate();
        while let Some((i, atom1)) = nodes.next() {
            for (j, atom2) in nodes.clone() {
                assert!(i < j);
                let mut weight = 0;
                for v1 in atom1.vars() {
                    for v2 in atom2.vars() {
                        if v1 == v2 {
                            weight += 1;
                        }
                    }
                }
                edges.push((weight, i, j));
            }
        }
        edges.sort();

        let n = self.atoms.len();
        let mut uf = UnionFind::new(n);

        for (_weight, i, j) in edges.iter().cloned().rev() {
            uf.union(i, j);
        }

        // everything is in the same set
        debug_assert!((0..n - 1).all(|i| uf.find(i) == uf.find(i + 1)));

        let tree = &uf.tree[uf.find(0)];
        self.compile_rec(tree)
    }

    fn compile_rec(&self, tree: &Tree) -> (VarMap<V>, Expr<S, T>) {
        match tree {
            Tree::Leaf((), i) => {
                let atom = &self.atoms[*i];
                let mut used_vars: FxHashMap<V, Vec<usize>> = Default::default();
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

                let expr = Expr::Scan {
                    relation: (atom.symbol.clone(), atom.arity),
                    var_eqs,
                    term_eqs,
                };
                (var_map, expr)
            }
            Tree::Node(_, _, tree1, tree2) => {
                let (v1, e1) = self.compile_rec(tree1);
                let (v2, e2) = self.compile_rec(tree2);
                let (key1, key2): (Vec<usize>, Vec<usize>) = v1
                    .iter()
                    .filter_map(|(v, i1)| v2.get(v).map(|i2| (i1, i2)))
                    .unzip();

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
                    }
                }

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
                (var_map, expr)
            }
        }
    }
}

struct UnionFind {
    parent: Vec<usize>,
    tree: Vec<Rc<Tree>>,
}

#[derive(Debug, Clone)]
enum Tree<T = ()> {
    Leaf(T, usize),
    Node(T, usize, Rc<Tree<T>>, Rc<Tree<T>>),
}

impl Tree {
    fn depth(&self) -> usize {
        match self {
            Tree::Leaf(_, _) => 1,
            Tree::Node(_, depth, _, _) => *depth,
        }
    }
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            tree: (0..n).map(|i| Rc::new(Tree::Leaf((), i))).collect(),
        }
    }

    fn find(&self, mut i: usize) -> usize {
        while i != self.parent[i] {
            i = self.parent[i];
        }
        i
    }

    fn union(&mut self, i: usize, j: usize) -> bool {
        let i = self.find(i);
        let j = self.find(j);
        if i == j {
            false
        } else {
            let i_depth = self.tree[i].depth();
            let j_depth = self.tree[j].depth();
            let new_tree = Rc::new(Tree::Node(
                (),
                i_depth.max(j_depth) + 1,
                self.tree[i].clone(),
                self.tree[j].clone(),
            ));
            if i_depth < j_depth {
                self.parent[i] = j;
                self.tree[j] = new_tree
            } else {
                self.parent[j] = i;
                self.tree[i] = new_tree;
            }
            true
        }
    }
}
