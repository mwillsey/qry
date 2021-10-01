#[derive(Debug, PartialEq, PartialOrd)]
pub struct Total<T>(pub T);

impl<T: PartialOrd> Ord for Total<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).expect("this can't fail")
    }
}

impl<T: PartialEq> Eq for Total<T> {}
