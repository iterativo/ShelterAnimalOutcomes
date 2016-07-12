import numpy

class DataSet(object):
    
    def __init__(self,
                 samples,
                 labels,
                 num_samples):
        """
        Construct a DataSet
        """
        self._samples = samples
        self._labels = labels
        self._index_in_epoch = 0
        self._num_samples = num_samples
        self._epochs_completed = 0
                 
    def next_batch(self, batch_size):
        """
        Return the next `batch_size` samples
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_samples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arrange(self._num_samples)
            numpy.random.suffle(perm)
            self._samples = self._samples[perm]
            # Start new epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_samples
        end = self._index_in_epoch
        return self._samples[start:end], self._labels[start:end]
            