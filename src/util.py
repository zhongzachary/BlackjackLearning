class NestedDict(dict):
    def __getitem__(self, keys) -> object:
        """
        >>> nd = NestedDict({'10': {'2': 3}})
        >>> nd['10','2']
        3
        >>> nd['10'] = {'2': {'3': 4}}
        >>> nd['10']
        {'2': {'3': 4}}
        >>> nd['10','2']
        {'3': 4}
        >>> nd['10','2','3']
        4
        """
        if not isinstance(keys, tuple):
            return dict.__getitem__(self, keys)
        curr_dict = self
        for k in keys:
            curr_dict = dict.__getitem__(curr_dict, k)
        return curr_dict

    def __setitem__(self, keys, value):
        """
        >>> nd = NestedDict()
        >>> nd['10'] = 2
        >>> nd['10']
        2
        >>> nd['10','3'] = 2
        >>> nd['10','3']
        2
        >>> nd['10','4'] = 5
        >>> nd['10','3']
        2
        >>> nd['10','4']
        5
        >>> nd['10','4','5'] = 10
        >>> nd['10','3']
        2
        >>> nd['10','4']
        {'5': 10}
        >>> nd['10','4','5']
        10
        """
        if not isinstance(keys, tuple):
            dict.__setitem__(self, keys, value)
        curr_dict = self
        for k in keys[:len(keys) - 1]:
            tmp_dict = curr_dict.setdefault(k, NestedDict())
            if isinstance(tmp_dict, dict):
                curr_dict = tmp_dict
            else:
                dict.__setitem__(curr_dict, k, NestedDict())
                curr_dict = curr_dict[k]
        dict.__setitem__(curr_dict, keys[len(keys) - 1], value)

    def contains(self, *keys):
        """
        >>> nd = NestedDict()
        >>> nd.contains('1','2')
        False
        >>> nd['1'] = {'2': {'3': 4}}
        >>> nd.contains('1')
        True
        >>> nd.contains('1', '2')
        True
        >>> nd.contains('1', '3')
        False
        >>> nd.contains('2')
        False
        >>> nd.contains('1', '2', '3')
        True
        """
        curr_dict = self
        for k in keys:
            if k in curr_dict:
                curr_dict = curr_dict[k]
            else:
                return False
        return True
