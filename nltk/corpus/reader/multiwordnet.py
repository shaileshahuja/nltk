# -*- coding: utf-8 -*-


from nltk.corpus.reader import CorpusReader
from collections import defaultdict
from nltk.corpus.util import LazyCorpusLoader
import nltk
import os
from wordnet_utils import *
from wordnet_classes import *
from englishreader import WordNetCorpusReader
from nltk import FreqDist


class MultiLinguilCorpusReader(CorpusReader):
    """
    A corpus reader used to access multiple WordNets
    from the Open Multilingual WordNet corpus and
    NLTK's english WordNet.
    """

    def __init__(self, root, lang, extended_wordnet):
        """
        Construct a new wordnet corpus reader, with the given root
        directory.
        """

        self._lang = lang
        self._data_file_name = 'wn-data-%s.tab' % self._lang
        super(MultiLinguilCorpusReader, self).__init__(root, self._data_file_name)

        self._extended_wordnet = extended_wordnet
        self._lemma_pos_offset_map = defaultdict(dict)
        """A index that provides the file offset

        Map from lemma -> pos -> synset_index -> offset"""

        self._synset_pos_lemma_map = defaultdict(dict)
        """A index that provides the lemmas

        Map from offset -> pos -> lemmas """

        self._data_file_stream = None
        # self._key_count_file = None
        # self._key_synset_file = None

        # Load the indices for lemmas and synset offsets
        self._load_lemma_pos_offset_map()

    def _load_lemma_pos_offset_map(self):
        # parse each line of the file (ignoring comment lines)

        for i, line in enumerate(self.open(self._data_file_name)):
            if line.startswith('#'):
                continue

            _iter = iter(line.split('\t'))
            _next_token = lambda: next(_iter)
            try:

                # get the offset and part-of-speech
                offset, pos = _next_token().split('-')
                offset = int(offset)
                # get the type of wordnet entity
                _ = _next_token()

                # get lemma name
                lemma = _next_token().strip()

            # raise more informative error with file name and line number
            except (AssertionError, ValueError) as e:
                tup = self._data_file_name, (i + 1), e
                raise WordNetError('file %s, line %i: %s' % tup)

            # map lemmas and parts of speech to synsets
            if pos not in self._lemma_pos_offset_map[lemma]:
                self._lemma_pos_offset_map[lemma][pos] = []
            self._lemma_pos_offset_map[lemma][pos].append(offset)
            if pos not in self._synset_pos_lemma_map[offset]:
                self._synset_pos_lemma_map[offset][pos] = []
            self._synset_pos_lemma_map[offset][pos].append(lemma)

    def synsets(self, lemma, pos=None):
        """Load all synsets with a given lemma and part of speech tag.
        If no pos is specified, all synsets for all parts of speech
        will be loaded.
        """
        lemma = lemma.lower()
        get_synset = self._synset_from_pos_and_offset
        index = self._lemma_pos_offset_map
        if lemma not in index:
            error = "Enter exact lemma name as morphological substitutions are not supported"
            raise WordNetError('Lemma %s not found. %s.' % (lemma, error))

        if pos is None:
            pos = POS_LIST

        return [get_synset(p, offset)
                for p in pos
                for offset in index[lemma].get(p, [])]

    def _synset_from_pos_and_offset(self, pos, offset):
        language_map = self._extended_wordnet._language_map['eng']
        synset = language_map._synset_from_pos_and_offset(pos, offset)
        synset.translate_lemma_names(self._lang)
        return synset

    #////////////////////////////////////////////////////////////
    # Loading Synsets
    #////////////////////////////////////////////////////////////
    def synset(self, name):
        synset = self._extended_wordnet._language_map['eng'].synset(name)
        synset.translate_lemma_names(self._lang)
        return synset

    def lemmas(self, lemma, pos=None):
        """Return all Lemma objects with a name matching the specified lemma
        name and part of speech tag. Matches any part of speech tag if none is
        specified."""
        lemma = lemma.lower()
        synsets = self.synsets(lemma, pos)
        return [lemma_obj
                for synset in synsets
                for lemma_obj in synset.lemmas[self._lang]
                if lemma_obj.name.lower() == lemma]

    def all_lemma_names(self, pos=None):
        """Return all lemma names for all synsets for the given
        part of speech tag. If pos is not specified, all synsets
        for all parts of speech will be used.
        """
        if pos is None:
            return iter(self._lemma_pos_offset_map)
        else:
            return (lemma
                    for lemma in self._lemma_pos_offset_map
                    if pos in self._lemma_pos_offset_map[lemma])

    def all_synsets(self, pos=None):
        """Iterate over all synsets with a given part of speech tag.
        If no pos is specified, all synsets for all parts of speech
        will be loaded.
        """
        synsets = self._extended_wordnet._language_map['eng'].synsets(pos)
        for synset in synsets:
            synset.translate_lemma_names(self._lang)
        return synsets

    def lemma_count(self, lemma):
        """Return the frequency count for this Lemma"""
        # open the count file if we haven't already
        return sum(len(self._lemma_pos_offset_map[lemma][pos])
                   for pos in self._lemma_pos_offset_map[lemma])

    def lemma(self, name):
        synset_name, lemma_name = name.rsplit('.', 1)
        synset = self.synset(synset_name)
        for lemma in synset.lemmas[self._lang]:
            if lemma.name == lemma_name:
                return lemma
        raise WordNetError('no lemma %r in %r' % (lemma_name, synset_name))

    def lemma_from_key(self, key):
        raise WordNetError("Method not yet supported for MultiLinguil Corpus")

    def get_version(self):
        raise WordNetError("Method not yet supported for MultiLinguil Corpus")

    def morphy(self, form, pos=None):
        raise WordNetError("Method not yet supported for MultiLinguil Corpus")

    def path_similarity(self, synset1, synset2, verbose=False, simulate_root=True):
        return synset1.path_similarity(synset2, verbose, simulate_root)
    path_similarity.__doc__ = Synset.path_similarity.__doc__

    def lch_similarity(self, synset1, synset2, verbose=False, simulate_root=True):
        return synset1.lch_similarity(synset2, verbose, simulate_root)
    lch_similarity.__doc__ = Synset.lch_similarity.__doc__

    def wup_similarity(self, synset1, synset2, verbose=False, simulate_root=True):
        return synset1.wup_similarity(synset2, verbose, simulate_root)
    wup_similarity.__doc__ = Synset.wup_similarity.__doc__

    def res_similarity(self, synset1, synset2, ic, verbose=False):
        return synset1.res_similarity(synset2, ic, verbose)
    res_similarity.__doc__ = Synset.res_similarity.__doc__

    def jcn_similarity(self, synset1, synset2, ic, verbose=False):
        return synset1.jcn_similarity(synset2, ic, verbose)
    jcn_similarity.__doc__ = Synset.jcn_similarity.__doc__

    def lin_similarity(self, synset1, synset2, ic, verbose=False):
        return synset1.lin_similarity(synset2, ic, verbose)
    lin_similarity.__doc__ = Synset.lin_similarity.__doc__

    def ic(self, corpus, weight_senses_equally=False, smoothing=1.0):
        """
        Creates an information content lookup dictionary from a corpus.

        :type corpus: CorpusReader
        :param corpus: The corpus from which we create an information
        content dictionary.
        :type weight_senses_equally: bool
        :param weight_senses_equally: If this is True, gives all
        possible senses equal weight rather than dividing by the
        number of possible senses.  (If a word has 3 synses, each
        sense gets 0.3333 per appearance when this is False, 1.0 when
        it is true.)
        :param smoothing: How much do we smooth synset counts (default is 1.0)
        :type smoothing: float
        :return: An information content dictionary
        """
        counts = FreqDist()
        for ww in corpus.words():
            counts.inc(ww)

        ic = {}
        for pp in POS_LIST:
            ic[pp] = defaultdict(float)

        # Initialize the counts with the smoothing value
        if smoothing > 0.0:
            for ss in self.all_synsets():
                pos = ss.pos
                if pos == ADJ_SAT:
                    pos = ADJ
                ic[pos][ss.offset] = smoothing

        for ww in counts:
            possible_synsets = self.synsets(ww)
            if len(possible_synsets) == 0:
                continue

            # Distribute weight among possible synsets
            weight = float(counts[ww])
            if not weight_senses_equally:
                weight /= float(len(possible_synsets))

            for ss in possible_synsets:
                pos = ss.pos
                if pos == ADJ_SAT:
                    pos = ADJ
                for level in ss._iter_hypernym_lists():
                    for hh in level:
                        ic[pos][hh.offset] += weight
                        # Add the weight to the root
                ic[pos][0] += weight
        return ic


class ExtendedWordNetCorpusReader(object):
    """
    A proxy class used to access the extended wordnet reader.
    """
    def __init__(self):
        """
        Construct a new wordnet corpus reader, with the given root
        directory. Uses the default WordNetCorpusReader for english
        and ExtendedWordNetCorpusReader for other languages.
        """
        self._language_map = dict()
        main_dir = nltk.data.find('corpora/multiwordnet')
        self._language_map['eng'] = LazyCorpusLoader('wordnet', WordNetCorpusReader, self)
        for lang in os.walk(main_dir).next()[1]:
            self._language_map[lang] = LazyCorpusLoader('multiwordnet/%s' % lang,
                                                        MultiLinguilCorpusReader,
                                                        lang, self)

    def __getattr__(self, attr):
        self._method = attr
        return self

    def __call__(self, *args, **kwargs):
        lang = kwargs.pop('lang', 'eng')
        assert isinstance(lang, str)
        assert lang in self._language_map
        return getattr(self._language_map[lang], self._method, None)(*args, **kwargs)


if __name__ == "__main__":
    ewn = ExtendedWordNetCorpusReader()

    dog = ewn.synset('dog.n.01')
    print dog.lemma_names
    print dog

    dogjpn = ewn.synset('dog.n.01', lang='jpn')
    print dog.lemma_names
    print dog.translate_lemma_names(['eng', 'fre', 'ind'])

    # Print lemmas from french wordnet
    print ewn.lemmas('vis', lang='fre')

    # Print lemmas from Japanese wordnet, tested on 2.7.3 only
    synsets = ewn.synsets(u'犬', lang='jpn')
    for l in synsets[0].lemmas['jpn']:
        print l.name  # prints japanese correctly

    print synsets[0].hypernym_paths()
