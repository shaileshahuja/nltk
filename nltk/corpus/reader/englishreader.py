from __future__ import print_function, unicode_literals

import re
from itertools import islice, chain
from collections import defaultdict
from wordnet_classes import *
from nltk.corpus.reader import CorpusReader
from nltk.util import binary_search_file as _binary_search_file
from nltk.probability import FreqDist
from nltk.compat import xrange

######################################################################
## WordNet Corpus Reader
######################################################################


class WordNetCorpusReader(CorpusReader):
    """
    A corpus reader used to access wordnet or its variants.
    """

    _ENCODING = 'utf8'

    #{ Part-of-speech constants
    ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
    #}

    #{ Filename constants
    _FILEMAP = {ADJ: 'adj', ADV: 'adv', NOUN: 'noun', VERB: 'verb'}
    #}

    #{ Part of speech constants
    _pos_numbers = {NOUN: 1, VERB: 2, ADJ: 3, ADV: 4, ADJ_SAT: 5}
    _pos_names = dict(tup[::-1] for tup in _pos_numbers.items())
    #}

    #: A list of file identifiers for all the fileids used by this
    #: corpus reader.
    _FILES = ('cntlist.rev', 'lexnames', 'index.sense',
              'index.adj', 'index.adv', 'index.noun', 'index.verb',
              'data.adj', 'data.adv', 'data.noun', 'data.verb',
              'adj.exc', 'adv.exc', 'noun.exc', 'verb.exc', )

    def __init__(self, root, extended_wordnet):
        """
        Construct a new wordnet corpus reader, with the given root
        directory.
        """
        super(WordNetCorpusReader, self).__init__(root, self._FILES,
                                                  encoding=self._ENCODING)

        self._extended_wordnet = extended_wordnet

        self._lemma_pos_offset_map = defaultdict(dict)
        """A index that provides the file offset

        Map from lemma -> pos -> synset_index -> offset"""

        self._synset_offset_cache = defaultdict(dict)
        """A cache so we don't have to reconstuct synsets

        Map from pos -> offset -> synset"""

        self._max_depth = defaultdict(dict)
        """A lookup for the maximum depth of each part of speech.  Useful for
        the lch similarity metric.
        """

        self._data_file_map = {}
        self._exception_map = {}
        self._lexnames = []
        self._key_count_file = None
        self._key_synset_file = None

        # Load the lexnames
        for i, line in enumerate(self.open('lexnames')):
            index, lexname, _ = line.split()
            assert int(index) == i
            self._lexnames.append(lexname)

        # Load the indices for lemmas and synset offsets
        self._load_lemma_pos_offset_map()

        # load the exception file data into memory
        self._load_exception_map()


    def _load_lemma_pos_offset_map(self):
        for suffix in self._FILEMAP.values():

            # parse each line of the file (ignoring comment lines)
            for i, line in enumerate(self.open('index.%s' % suffix)):
                if line.startswith(' '):
                    continue

                _iter = iter(line.split())
                _next_token = lambda: next(_iter)
                try:

                    # get the lemma and part-of-speech
                    lemma = _next_token()
                    pos = _next_token()

                    # get the number of synsets for this lemma
                    n_synsets = int(_next_token())
                    assert n_synsets > 0

                    # get the pointer symbols for all synsets of this lemma
                    n_pointers = int(_next_token())
                    _ = [_next_token() for _ in xrange(n_pointers)]

                    # same as number of synsets
                    n_senses = int(_next_token())
                    assert n_synsets == n_senses

                    # get number of senses ranked according to frequency
                    _ = int(_next_token())

                    # get synset offsets
                    synset_offsets = [int(_next_token()) for _ in xrange(n_synsets)]

                # raise more informative error with file name and line number
                except (AssertionError, ValueError) as e:
                    tup = ('index.%s' % suffix), (i + 1), e
                    raise WordNetError('file %s, line %i: %s' % tup)

                # map lemmas and parts of speech to synsets
                self._lemma_pos_offset_map[lemma][pos] = synset_offsets
                if pos == ADJ:
                    self._lemma_pos_offset_map[lemma][ADJ_SAT] = synset_offsets

    def _load_exception_map(self):
        # load the exception file data into memory
        for pos, suffix in self._FILEMAP.items():
            self._exception_map[pos] = {}
            for line in self.open('%s.exc' % suffix):
                terms = line.split()
                self._exception_map[pos][terms[0]] = terms[1:]
        self._exception_map[ADJ_SAT] = self._exception_map[ADJ]

    def _compute_max_depth(self, pos, simulate_root):
        """
        Compute the max depth for the given part of speech.  This is
        used by the lch similarity metric.
        """
        depth = 0
        for ii in self.all_synsets(pos):
            try:
                depth = max(depth, ii.max_depth())
            except RuntimeError:
                print(ii)
        if simulate_root:
            depth += 1
        self._max_depth[pos] = depth

    def get_version(self):
        fh = self._data_file(ADJ)
        for line in fh:
            match = re.search(r'WordNet (\d+\.\d+) Copyright', line)
            if match is not None:
                version = match.group(1)
                fh.seek(0)
                return version

    #////////////////////////////////////////////////////////////
    # Loading Lemmas
    #////////////////////////////////////////////////////////////
    def lemma(self, name):
        synset_name, lemma_name = name.rsplit('.', 1)
        synset = self.synset(synset_name)
        for lemma in synset.lemmas:
            if lemma.name == lemma_name:
                return lemma
        raise WordNetError('no lemma %r in %r' % (lemma_name, synset_name))

    def lemma_from_key(self, key):
        # Keys are case sensitive and always lower-case
        key = key.lower()

        lemma_name, lex_sense = key.split('%')
        pos_number, lexname_index, lex_id, _, _ = lex_sense.split(':')
        pos = self._pos_names[int(pos_number)]

        # open the key -> synset file if necessary
        if self._key_synset_file is None:
            self._key_synset_file = self.open('index.sense')

        # Find the synset for the lemma.
        synset_line = _binary_search_file(self._key_synset_file, key)
        if not synset_line:
            raise WordNetError("No synset found for key %r" % key)
        offset = int(synset_line.split()[1])
        synset = self._synset_from_pos_and_offset(pos, offset)

        # return the corresponding lemma
        for lemma in synset.lemmas:
            if lemma.key == key:
                return lemma
        raise WordNetError("No lemma found for for key %r" % key)

    #////////////////////////////////////////////////////////////
    # Loading Synsets
    #////////////////////////////////////////////////////////////
    def synset(self, name):
        # split name into lemma, part of speech and synset number
        lemma, pos, synset_index_str = name.lower().rsplit('.', 2)
        synset_index = int(synset_index_str) - 1

        # get the offset for this synset
        try:
            offset = self._lemma_pos_offset_map[lemma][pos][synset_index]
        except KeyError:
            message = 'no lemma %r with part of speech %r'
            raise WordNetError(message % (lemma, pos))
        except IndexError:
            n_senses = len(self._lemma_pos_offset_map[lemma][pos])
            message = "lemma %r with part of speech %r has only %i %s"
            if n_senses == 1:
                tup = lemma, pos, n_senses, "sense"
            else:
                tup = lemma, pos, n_senses, "senses"
            raise WordNetError(message % tup)

        # load synset information from the appropriate file
        synset = self._synset_from_pos_and_offset(pos, offset)

        # some basic sanity checks on loaded attributes
        if pos == 's' and synset.pos == 'a':
            message = ('adjective satellite requested but only plain '
                       'adjective found for lemma %r')
            raise WordNetError(message % lemma)
        assert synset.pos == pos or (pos == 'a' and synset.pos == 's')

        # Return the synset object.
        return synset

    def _data_file(self, pos):
        """
        Return an open file pointer for the data file for the given
        part of speech.
        """
        if pos == ADJ_SAT:
            pos = ADJ
        if self._data_file_map.get(pos) is None:
            fileid = 'data.%s' % self._FILEMAP[pos]
            self._data_file_map[pos] = self.open(fileid)
        return self._data_file_map[pos]

    def _synset_from_pos_and_offset(self, pos, offset):
        # Check to see if the synset is in the cache
        if offset in self._synset_offset_cache[pos]:
            return self._synset_offset_cache[pos][offset]

        data_file = self._data_file(pos)
        data_file.seek(offset)
        data_file_line = data_file.readline()
        synset = self._synset_from_pos_and_line(pos, data_file_line)
        assert synset.offset == offset
        self._synset_offset_cache[pos][offset] = synset
        return synset

    def _synset_from_pos_and_line(self, pos, data_file_line):
        # Construct a new (empty) synset.
        synset = Synset(self, self._extended_wordnet)

        # parse the entry for this synset
        try:

            # parse out the definitions and examples from the gloss
            columns_str, gloss = data_file_line.split('|')
            gloss = gloss.strip()
            definitions = []
            for gloss_part in gloss.split(';'):
                gloss_part = gloss_part.strip()
                if gloss_part.startswith('"'):
                    synset.examples.append(gloss_part.strip('"'))
                else:
                    definitions.append(gloss_part)
            synset.definition = '; '.join(definitions)

            # split the other info into fields
            _iter = iter(columns_str.split())
            _next_token = lambda: next(_iter)

            # get the offset
            synset.offset = int(_next_token())

            # determine the lexicographer file name
            lexname_index = int(_next_token())
            synset.lexname = self._lexnames[lexname_index]

            # get the part of speech
            synset.pos = _next_token()

            # create Lemma objects for each lemma
            n_lemmas = int(_next_token(), 16)
            for _ in xrange(n_lemmas):
                # get the lemma name
                lemma_name = _next_token()
                # get the lex_id (used for sense_keys)
                lex_id = int(_next_token(), 16)
                # If the lemma has a syntactic marker, extract it.
                m = re.match(r'(.*?)(\(.*\))?$', lemma_name)
                lemma_name, syn_mark = m.groups()
                # create the lemma object
                lemma = Lemma(self, synset, lemma_name, lexname_index,
                              lex_id, syn_mark)
                synset.lemmas['eng'].append(lemma)
                synset.lemma_names['eng'].append(lemma.name)

            # collect the pointer tuples
            n_pointers = int(_next_token())
            for _ in xrange(n_pointers):
                symbol = _next_token()
                offset = int(_next_token())
                pos = _next_token()
                lemma_ids_str = _next_token()
                if lemma_ids_str == '0000':
                    synset._pointers[symbol].add((pos, offset))
                else:
                    source_index = int(lemma_ids_str[:2], 16) - 1
                    target_index = int(lemma_ids_str[2:], 16) - 1
                    source_lemma_name = synset.lemmas['eng'][source_index].name
                    lemma_pointers = synset._lemma_pointers
                    tups = lemma_pointers[source_lemma_name, symbol]
                    tups.add((pos, offset, target_index))

            # read the verb frames
            try:
                frame_count = int(_next_token())
            except StopIteration:
                pass
            else:
                for _ in xrange(frame_count):
                    # read the plus sign
                    plus = _next_token()
                    assert plus == '+'
                    # read the frame and lemma number
                    frame_number = int(_next_token())
                    frame_string_fmt = VERB_FRAME_STRINGS[frame_number]
                    lemma_number = int(_next_token(), 16)
                    # lemma number of 00 means all words in the synset
                    if lemma_number == 0:
                        synset.frame_ids.append(frame_number)
                        for lemma in synset.lemmas['eng']:
                            lemma.frame_ids.append(frame_number)
                            lemma.frame_strings.append(frame_string_fmt %
                                                       lemma.name)
                    # only a specific word in the synset
                    else:
                        lemma = synset.lemmas['eng'][lemma_number - 1]
                        lemma.frame_ids.append(frame_number)
                        lemma.frame_strings.append(frame_string_fmt %
                                                   lemma.name)

        # raise a more informative error with line text
        except ValueError as e:
            raise WordNetError('line %r: %s' % (data_file_line, e))

        # set sense keys for Lemma objects - note that this has to be
        # done afterwards so that the relations are available
        for lemma in synset.lemmas['eng']:
            if synset.pos == ADJ_SAT:
                head_lemma = synset.similar_tos()[0].lemmas[0]
                head_name = head_lemma.name
                head_id = '%02d' % head_lemma._lex_id
            else:
                head_name = head_id = ''
            tup = (lemma.name, WordNetCorpusReader._pos_numbers[synset.pos],
                   lemma._lexname_index, lemma._lex_id, head_name, head_id)
            lemma.key = ('%s%%%d:%02d:%02d:%s:%s' % tup).lower()

        # the canonical name is based on the first lemma
        lemma_name = synset.lemmas['eng'][0].name.lower()
        offsets = self._lemma_pos_offset_map[lemma_name][synset.pos]
        sense_index = offsets.index(synset.offset)
        tup = lemma_name, synset.pos, sense_index + 1
        synset.name = '%s.%s.%02i' % tup

        return synset

    #////////////////////////////////////////////////////////////
    # Retrieve synsets and lemmas.
    #////////////////////////////////////////////////////////////
    def synsets(self, lemma, pos=None, lang='eng'):
        """Load all synsets with a given lemma and part of speech tag.
        If no pos is specified, all synsets for all parts of speech
        will be loaded.
        """
        lemma = lemma.lower()
        get_synset = self._synset_from_pos_and_offset
        index = self._lemma_pos_offset_map

        if pos is None:
            pos = POS_LIST

        return [get_synset(p, offset)
                for p in pos
                for form in self._morphy(lemma, p)
                for offset in index[form].get(p, [])]

    def lemmas(self, lemma, pos=None):
        """Return all Lemma objects with a name matching the specified lemma
        name and part of speech tag. Matches any part of speech tag if none is
        specified."""
        lemma = lemma.lower()
        return [lemma_obj
                for synset in self.synsets(lemma, pos)
                for lemma_obj in synset.lemmas
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
        if pos is None:
            pos_tags = self._FILEMAP.keys()
        else:
            pos_tags = [pos]

        cache = self._synset_offset_cache
        from_pos_and_line = self._synset_from_pos_and_line

        # generate all synsets for each part of speech
        for pos_tag in pos_tags:
            # Open the file for reading.  Note that we can not re-use
            # the file poitners from self._data_file_map here, because
            # we're defining an iterator, and those file pointers might
            # be moved while we're not looking.
            if pos_tag == ADJ_SAT:
                pos_tag = ADJ
            fileid = 'data.%s' % self._FILEMAP[pos_tag]
            data_file = self.open(fileid)

            try:
                # generate synsets for each line in the POS file
                offset = data_file.tell()
                line = data_file.readline()
                while line:
                    if not line[0].isspace():
                        if offset in cache[pos_tag]:
                            # See if the synset is cached
                            synset = cache[pos_tag][offset]
                        else:
                            # Otherwise, parse the line
                            synset = from_pos_and_line(pos_tag, line)
                            cache[pos_tag][offset] = synset

                        # adjective satellites are in the same file as
                        # adjectives so only yield the synset if it's actually
                        # a satellite
                        if pos_tag == ADJ_SAT:
                            if synset.pos == pos_tag:
                                yield synset

                        # for all other POS tags, yield all synsets (this means
                        # that adjectives also include adjective satellites)
                        else:
                            yield synset
                    offset = data_file.tell()
                    line = data_file.readline()

            # close the extra file handle we opened
            except:
                data_file.close()
                raise
            else:
                data_file.close()

    #////////////////////////////////////////////////////////////
    # Misc
    #////////////////////////////////////////////////////////////
    def lemma_count(self, lemma):
        """Return the frequency count for this Lemma"""
        # open the count file if we haven't already
        if self._key_count_file is None:
            self._key_count_file = self.open('cntlist.rev')
            # find the key in the counts file and return the count
        line = _binary_search_file(self._key_count_file, lemma.key)
        if line:
            return int(line.rsplit(' ', 1)[-1])
        else:
            return 0

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

    #////////////////////////////////////////////////////////////
    # Morphy
    #////////////////////////////////////////////////////////////
    # Morphy, adapted from Oliver Steele's pywordnet
    def morphy(self, form, pos=None):
        """
        Find a possible base form for the given form, with the given
        part of speech, by checking WordNet's list of exceptional
        forms, and by recursively stripping affixes for this part of
        speech until a form in WordNet is found.

        >>> from nltk.corpus import wordnet as wn
        >>> print(wn.morphy('dogs'))
        dog
        >>> print(wn.morphy('churches'))
        church
        >>> print(wn.morphy('aardwolves'))
        aardwolf
        >>> print(wn.morphy('abaci'))
        abacus
        >>> wn.morphy('hardrock', wn.ADV)
        >>> print(wn.morphy('book', wn.NOUN))
        book
        >>> wn.morphy('book', wn.ADJ)
        """

        if pos is None:
            morphy = self._morphy
            analyses = chain(a for p in POS_LIST for a in morphy(form, p))
        else:
            analyses = self._morphy(form, pos)

        # get the first one we find
        first = list(islice(analyses, 1))
        if len(first) == 1:
            return first[0]
        else:
            return None

    MORPHOLOGICAL_SUBSTITUTIONS = {
        NOUN: [('s', ''), ('ses', 's'), ('ves', 'f'), ('xes', 'x'),
               ('zes', 'z'), ('ches', 'ch'), ('shes', 'sh'),
               ('men', 'man'), ('ies', 'y')],
        VERB: [('s', ''), ('ies', 'y'), ('es', 'e'), ('es', ''),
               ('ed', 'e'), ('ed', ''), ('ing', 'e'), ('ing', '')],
        ADJ: [('er', ''), ('est', ''), ('er', 'e'), ('est', 'e')],
        ADV: []}

    def _morphy(self, form, pos):
        # from jordanbg:
        # Given an original string x
        # 1. Apply rules once to the input to get y1, y2, y3, etc.
        # 2. Return all that are in the database
        # 3. If there are no matches, keep applying rules until you either
        #    find a match or you can't go any further

        exceptions = self._exception_map[pos]
        substitutions = self.MORPHOLOGICAL_SUBSTITUTIONS[pos]

        def apply_rules(forms):
            return [form[:-len(old)] + new
                    for form in forms
                    for old, new in substitutions
                    if form.endswith(old)]

        def filter_forms(forms):
            result = []
            seen = set()
            for form in forms:
                if form in self._lemma_pos_offset_map:
                    if pos in self._lemma_pos_offset_map[form]:
                        if form not in seen:
                            result.append(form)
                            seen.add(form)
            return result

        # 0. Check the exception lists
        if form in exceptions:
            return filter_forms([form] + exceptions[form])

        # 1. Apply rules once to the input to get y1, y2, y3, etc.
        forms = apply_rules([form])

        # 2. Return all that are in the database (and check the original too)
        results = filter_forms([form] + forms)
        if results:
            return results

        # 3. If there are no matches, keep applying rules until we find a match
        while forms:
            forms = apply_rules(forms)
            results = filter_forms(forms)
            if results:
                return results

        # Return an empty list if we can't find anything
        return []

    #////////////////////////////////////////////////////////////
    # Create information content from corpus
    #////////////////////////////////////////////////////////////
    def ic(self, corpus, weight_senses_equally = False, smoothing = 1.0):
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
