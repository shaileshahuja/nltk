import math

######################################################################
## Constants
######################################################################

#: Positive infinity (for similarity functions)
_INF = 1e300

#{ Part-of-speech constants
ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
#}

POS_LIST = [NOUN, VERB, ADJ, ADV]

#: A table of strings that are used to express verb frames.
VERB_FRAME_STRINGS = (
    None,
    "Something %s",
    "Somebody %s",
    "It is %sing",
    "Something is %sing PP",
    "Something %s something Adjective/Noun",
    "Something %s Adjective/Noun",
    "Somebody %s Adjective",
    "Somebody %s something",
    "Somebody %s somebody",
    "Something %s somebody",
    "Something %s something",
    "Something %s to somebody",
    "Somebody %s on something",
    "Somebody %s somebody something",
    "Somebody %s something to somebody",
    "Somebody %s something from somebody",
    "Somebody %s somebody with something",
    "Somebody %s somebody of something",
    "Somebody %s something on somebody",
    "Somebody %s somebody PP",
    "Somebody %s something PP",
    "Somebody %s PP",
    "Somebody's (body part) %s",
    "Somebody %s somebody to INFINITIVE",
    "Somebody %s somebody INFINITIVE",
    "Somebody %s that CLAUSE",
    "Somebody %s to somebody",
    "Somebody %s to INFINITIVE",
    "Somebody %s whether INFINITIVE",
    "Somebody %s somebody into V-ing something",
    "Somebody %s something with something",
    "Somebody %s INFINITIVE",
    "Somebody %s VERB-ing",
    "It %s that CLAUSE",
    "Something %s INFINITIVE")



def _lcs_by_depth(synset1, synset2, verbose=False):
    """
    Finds the least common subsumer of two synsets in a WordNet taxonomy,
    where the least common subsumer is defined as the ancestor node common
    to both input synsets whose shortest path to the root node is the longest.

    :type synset1: Synset
    :param synset1: First input synset.
    :type synset2: Synset
    :param synset2: Second input synset.
    :return: The ancestor synset common to both input synsets which is also the
    LCS.
    """
    subsumer = None
    max_min_path_length = -1

    subsumers = synset1.common_hypernyms(synset2)

    if verbose:
        print("> Subsumers1:", subsumers)

    # Eliminate those synsets which are ancestors of other synsets in the
    # set of subsumers.

    eliminated = set()
    hypernym_relation = lambda s: s.hypernyms() + s.instance_hypernyms()
    for s1 in subsumers:
        for s2 in subsumers:
            if s2 in s1.closure(hypernym_relation):
                eliminated.add(s2)
    if verbose:
        print("> Eliminated:", eliminated)

    subsumers = [s for s in subsumers if s not in eliminated]

    if verbose:
        print("> Subsumers2:", subsumers)

    # Calculate the length of the shortest path to the root for each
    # subsumer. Select the subsumer with the longest of these.

    for candidate in subsumers:

        paths_to_root = candidate.hypernym_paths()
        min_path_length = -1

        for path in paths_to_root:
            if min_path_length < 0 or len(path) < min_path_length:
                min_path_length = len(path)

        if min_path_length > max_min_path_length:
            max_min_path_length = min_path_length
            subsumer = candidate

    if verbose:
        print("> LCS Subsumer by depth:", subsumer)
    return subsumer


def _lcs_ic(synset1, synset2, ic, verbose=False):
    """
    Get the information content of the least common subsumer that has
    the highest information content value.  If two nodes have no
    explicit common subsumer, assume that they share an artificial
    root node that is the hypernym of all explicit roots.

    :type synset1: Synset
    :param synset1: First input synset.
    :type synset2: Synset
    :param synset2: Second input synset.  Must be the same part of
    speech as the first synset.
    :type  ic: dict
    :param ic: an information content object (as returned by ``load_ic()``).
    :return: The information content of the two synsets and their most
    informative subsumer
    """
    if synset1.pos != synset2.pos:
        raise WordNetError('Computing the least common subsumer requires ' + \
                           '%s and %s to have the same part of speech.' % \
                           (synset1, synset2))

    ic1 = information_content(synset1, ic)
    ic2 = information_content(synset2, ic)
    subsumers = synset1.common_hypernyms(synset2)
    if len(subsumers) == 0:
        subsumer_ic = 0
    else:
        subsumer_ic = max(information_content(s, ic) for s in subsumers)

    if verbose:
        print("> LCS Subsumer by content:", subsumer_ic)

    return ic1, ic2, subsumer_ic


# Utility functions

def information_content(synset, ic):
    try:
        icpos = ic[synset.pos]
    except KeyError:
        msg = 'Information content file has no entries for part-of-speech: %s'
        raise WordNetError(msg % synset.pos)

    counts = icpos[synset.offset]
    if counts == 0:
        return _INF
    else:
        return -math.log(counts / icpos[0])


# get the part of speech (NOUN or VERB) from the information content record
# (each identifier has a 'n' or 'v' suffix)

def _get_pos(field):
    if field[-1] == 'n':
        return NOUN
    elif field[-1] == 'v':
        return VERB
    else:
        msg = "Unidentified part of speech in WordNet Information Content file for field %s" % field
        raise ValueError(msg)


# unload corpus after tests
def teardown_module(module):
    from nltk.corpus import wordnet
    wordnet._unload()


class WordNetError(Exception):
    """An exception class for wordnet-related errors."""
