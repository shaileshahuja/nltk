# -*- coding: utf-8 -*-

__author__ = 'shailesh'

from nltk.corpus.reader import CorpusReader
from collections import defaultdict
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader.wordnet import WordNetCorpusReader, _WordNetObject, WordNetError
from nltk.compat import xrange, python_2_unicode_compatible
import nltk
import os

#{ Part-of-speech constants
ADJ, ADV, NOUN, VERB = 'a', 'r', 'n', 'v'
#}
POS_LIST = [NOUN, VERB, ADJ, ADV]


@python_2_unicode_compatible
class Lemma(_WordNetObject):
    """
    The lexical entry for a single morphological form of a
    sense-disambiguated word.

    Create a Lemma from a "<word>.<pos>.<number>.<lemma>" string where:
    <word> is the morphological stem identifying the synset
    <pos> is one of the module attributes ADJ, ADV, NOUN or VERB
    <number> is the sense number, counting from 0.

    Note that <word> and <lemma> can be different, e.g. the Synset
    'salt.n.03' has the Lemmas 'salt.n.03.salt', 'salt.n.03.saltiness' and
    'salt.n.03.salinity'.

    Lemma attributes:

    - name: The canonical name of this lemma.
    - synset: The synset that this lemma belongs to.

    Lemma methods:

    Lemmas have the following methods for retrieving related Lemmas. They
    correspond to the names for the pointer symbols defined here:
    http://wordnet.princeton.edu/man/wninput.5WN.html#sect3
    These methods all return lists of Lemmas:

    - antonyms
    - hypernyms, instance_hypernyms
    - hyponyms, instance_hyponyms
    - member_holonyms, substance_holonyms, part_holonyms
    - member_meronyms, substance_meronyms, part_meronyms
    - topic_domains, region_domains, usage_domains
    - attributes
    - derivationally_related_forms
    - entailments
    - causes
    - also_sees
    - verb_groups
    - similar_tos
    - pertainyms
    """

    __slots__ = ['_wordnet_corpus_reader', 'name', 'syntactic_marker',
                 'synset', 'frame_strings', 'frame_ids',
                 '_lexname_index', '_lex_id', 'key']

    # formerly _from_synset_info
    def __init__(self, wordnet_corpus_reader, synset, name):
        self._wordnet_corpus_reader = wordnet_corpus_reader
        self.name = name
        self.synset = synset
        self.frame_strings = []
        self.frame_ids = []

        self.key = None # gets set later.

    def __repr__(self):
        tup = type(self).__name__, self.synset.name, self.name
        return u"%s('%s.%s')" % tup

    def __str__(self):
        tup = type(self).__name__, self.synset.name, self.name
        return u"%s('%s.%s')" % tup


@python_2_unicode_compatible
class Synset(_WordNetObject):
    """Create a Synset from a "<lemma>.<pos>.<number>" string where:
    <lemma> is the word's morphological stem
    <pos> is one of the module attributes ADJ, ADV, NOUN or VERB

    Synset attributes:

    - name: The canonical name of this synset, formed using the first lemma
      of this synset. Note that this may be different from the name
      passed to the constructor if that string used a different lemma to
      identify the synset.
    - pos: The synset's part of speech, matching one of the module level
      attributes ADJ, ADJ_SAT, ADV, NOUN or VERB.
    - lemmas: A list of the Lemma objects for this synset.
    - definition: The definition for this synset.
    - offset: The offset in the WordNet dict file of this synset.

    Synset methods:

    Synsets have the following methods for retrieving related Synsets.
    They correspond to the names for the pointer symbols defined here:
    http://wordnet.princeton.edu/man/wninput.5WN.html#sect3
    These methods all return lists of Synsets.

    - hypernyms, instance_hypernyms
    - hyponyms, instance_hyponyms
    - member_holonyms, substance_holonyms, part_holonyms
    - member_meronyms, substance_meronyms, part_meronyms
    - attributes
    - entailments
    - causes
    - also_sees
    - verb_groups
    - similar_tos

    Additionally, Synsets support the following methods specific to the
    hypernym relation:

    - root_hypernyms
    - common_hypernyms
    - lowest_common_hypernyms

    Note that Synsets do not support the following relations because
    these are defined by WordNet as lexical relations:

    - antonyms
    - derivationally_related_forms
    - pertainyms
    """

    __slots__ = ['pos', 'offset', 'name', 'frame_ids',
                 'lemmas', 'lemma_names',
                 'definition', 'examples', 'lexname',
                 '_pointers', '_lemma_pointers', '_max_depth',
                 '_min_depth', ]

    def __init__(self, wordnet_corpus_reader):
        self._wordnet_corpus_reader = wordnet_corpus_reader
        # All of these attributes get initialized by
        # WordNetCorpusReader._synset_from_pos_and_line()

        self.pos = None
        self.offset = None
        self.name = None
        self.frame_ids = []
        self.lemmas = []
        self.lemma_names = []

        self._pointers = defaultdict(set)
        self._lemma_pointers = defaultdict(set)

    def __repr__(self):
        return u"%s('%s')" % (type(self).__name__, self.name)


class MultiLinguilCorpusReader(CorpusReader):

    def __init__(self, root, lang):
        """
        Construct a new wordnet corpus reader, with the given root
        directory.
        """

        self._lang = lang
        self._data_file_name = 'wn-data-%s.tab' % self._lang
        super(MultiLinguilCorpusReader, self).__init__(root, self._data_file_name)

        self._lemma_pos_offset_map = defaultdict(dict)
        """A index that provides the file offset

        Map from lemma -> pos -> synset_index -> offset"""

        self._synset_pos_lemma_map = defaultdict(dict)
        """A index that provides the lemmas

        Map from offset -> pos -> lemmas """
        self._synset_offset_cache = defaultdict(dict)
        """A cache so we don't have to reconstuct synsets

        Map from pos -> offset -> synset"""

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

            _iter = iter(line.split())
            _next_token = lambda: next(_iter)
            try:

                # get the offset and part-of-speech
                offset, pos = _next_token().split('-')
                offset = int(offset)
                # get the type of wordnet entity
                _ = _next_token()

                # get lemma name
                lemma = _next_token()

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
            error = "Enter exact lemma name as morphological substitutions are not supported yet for multiwordnet"
            raise WordNetError('Lemma %s not found. %s.' % (lemma, error))

        if pos is None:
            pos = POS_LIST

        return [get_synset(p, offset)
                for p in pos
                for offset in index[lemma].get(p, [])]

    def _synset_from_pos_and_offset(self, pos, offset):
        # Check to see if the synset is in the cache
        if offset in self._synset_offset_cache[pos]:
            return self._synset_offset_cache[pos][offset]

        synset_map = self._synset_pos_lemma_map
        synset = Synset(self)
        synset.offset = int(offset)
        synset.pos = pos
        for lemma_name in synset_map[offset][pos]:
            lemma = Lemma(self, synset, lemma_name)
            synset.lemmas.append(lemma)
            synset.lemma_names.append(lemma.name)

        # the canonical name is based on the first lemma
        lemma_name = synset.lemmas[0].name.lower()
        offsets = self._lemma_pos_offset_map[lemma_name][synset.pos]
        sense_index = offsets.index(synset.offset)
        tup = lemma_name, synset.pos, sense_index + 1
        synset.name = '%s.%s.%02i' % tup

        assert synset.offset == offset
        self._synset_offset_cache[pos][offset] = synset
        return synset


class MultiWordNetCorpusReader(object):
    """
    A corpus reader used to access wordnet or its variants.
    """

    def __init__(self):
        """
        Construct a new wordnet corpus reader, with the given root
        directory.
        """
        self._language_reader_map = dict()
        main_dir = nltk.data.find('corpora/multiwordnet')

        for lang in os.walk(main_dir).next()[1]:
            self._language_reader_map[lang] = LazyCorpusLoader('multiwordnet/%s' % lang, MultiLinguilCorpusReader,
                                                               lang)

    def synsets(self, lemma, pos=None, lang='eng'):
        assert isinstance(lang, str) or isinstance(lang, list)
        if isinstance(lang, str):
            return self._language_reader_map[lang].synsets(lemma, pos)
        if isinstance(lang, list):
            synsets = []
            for _lang in lang:
                    synsets.append(self._language_reader_map[_lang].synsets(lemma, pos))
            return synsets

    def lemmas(self, lemma, pos=None, lang='eng'):
        """Return all Lemma objects with a name matching the specified lemma
        name and part of speech tag. Matches any part of speech tag if none is
        specified."""
        assert isinstance(lang, str) or isinstance(lang, list)
        lemma = lemma.lower()
        synsets = self.synsets(lemma, pos, lang)
        if isinstance(lang, str):
            return [lemma_obj
                    for synset in synsets
                    for lemma_obj in synset.lemmas
                    if lemma_obj.name.lower() == lemma]
        if isinstance(lang, list):
            lemmas = []
            for synlist in synsets:
                lemmas.append([lemma_obj
                               for synset in synlist
                               for lemma_obj in synset.lemmas
                               if lemma_obj.name.lower() == lemma])
            return lemmas

if __name__ == "__main__":
    mwn = MultiWordNetCorpusReader()

    # Print lemmas from french and english wordnets
    print mwn.lemmas('vis', lang=['fre', 'eng'])

    # Print lemmas from Japanese wordnet, tested on 2.7.3 only
    lemmas = mwn.lemmas(u'犬', lang=['jpn'])
    for l in lemmas[0]:
        print l          # __repr__ doesn't return unicode
        print unicode(l) # prints japanese correctly
