def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """
    def __getitem__(self, index):
        data, target = cls[index]
        return data, target, index

    cls.__getitem__ = __getitem__
    return cls