# The following code is from openai/evolution-strategies-starter
# (https://github.com/openai/evolution-strategies-starter)
# under the MIT License.


import numpy as np


def _comp_ranks(x):
    """
    Returns ranks in [0, len(x))

    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """

    # init empty vector for return
    ranks = np.empty(len(x), dtype=int)

    # Compute ranks of x
    #  argsort() returns indices of ranks
    #  arange() returns a range 0:x
    #  This is the fastest way to do this.
    ranks[x.argsort()] = np.arange(len(x))

    return ranks


def compute_CS_ranks(x):
    '''Compute Centered and Scaled Ranks
    
    This function takes an object (presumbably the scores), centers the scores 
    on 0, and normalizes between -0.5 and 0.5, inclusive.

    Parameters
    ---------- 
    x: 2-D Numpy array
        This is a [`batch_size`*`batches_per_chunk`, 2] array of evaluation scores

    Returns
    -------
    y: Numpy array
        ?? More info?

    Reference
    ---------
    https://stats.stackexchange.com/questions/164833/how-do-i-normalize-a-vector-of-numbers-so-they-are-between-0-and-1

    '''

    # Safety check
    if x.size == 1:
        #print("stats.computer_centered_ranks:: x is of size 1")
        return(0)

    # Compute ranks
    #  object is the same dimensions as x
    #  has values [0, len(x))
    y = _comp_ranks(x.ravel()).reshape(x.shape).astype(np.float32)

    # Normalize to (0, 1)
    #  This works because the values are ranks, 0 to len(x)
    #  so max(x) - min(x) = size - 1
    y /= (x.size - 1)

    # Center vector at 0
    #  The vector is already normed (0,1), so shift it by -0.5,
    #  it's not centered at 0, with range (-0.5, 0.5)
    y -= .5

    return y


def compute_weighted_sum(weights, vec_generator, theta_size):
    ''' Calculates a weighted sum
    
    This function calculates a weighted sum using the weights input list as the value for 
    each vector from vec_generator. There is no safety if vec_generator is longer 
    than weights.

    Parameters
    ---------- 
    weights: list
        List of numeric weights
    vec_generator: generator
        generator of numpy arrays, where the generator is the same length as the weights
    theta_size: int
        Size of arrays created by vec_generator

    Returns
    ------- 
    total: np.Array
        Weighted sum of vectors, length theta_size
    num_items_summed: int
        Count of the items combined. This should be the same length as weights, but 
        is defined as the number of items from vec_generator
    '''

    # setup return objects
    #  use the summation as a counter
    total = np.zeros(theta_size, dtype=np.float32)
    num_items = 0

    # loop through noise generator
    for vec in vec_generator:
        # grab weight and multiply by noise
        total += (weights[num_items] * vec)
        # increment counter
        num_items += 1

    # return total vector and number of estimates combined
    return total, num_items



# def itergroups(items, group_size):
#     ''' A generator that takes a list of items and break it up into chunks of length group_size
    
#     Note: this is a generator, so it holds state. Each time it is called it passes through the code
#     until it hits a yeild and then returns that value. After the list is finished the generator is
#     exhausted.

#     Parameters
#     ---------- 
#     items: list
#         List of items to be broken into subgroups
#     group_size: int
#         Length of new list size
    
#     Yields
#     ------
#     tuple
#         tuple of items of size at most group_size
#     '''
#     assert group_size >= 1
#     group = []
#     for x in items:
#         group.append(x)
#         if len(group) == group_size:
#             yield tuple(group)
#             del group[:]
#     if group:
#         yield tuple(group)


# # this function performs a vector dot matrix operation to calculated the weighted sum.
# #  It first expands the vecs generator into a full list, then converts it into a matrix.
# #  This is very memory inefficent, and doesn't seem to offer performance benefits. 
# #  The "batch_size" limits the number of weights combined with vecs per loop iteration. 
# #  E.G., instead of a [1,len(weights)].dot([len(weights),len(theta)]) product, we 
# #  have a maximum size of [1,batch_size].dot([batch_size,len(theta)]) event. 
# #  This could have been for parallelization, but would still be quite memory heavy.
# #  Alternatively, if kept single-threaded, it would reduce memory usage, but it can't 
# #  be any more efficient than the generator already is. 
# #
# #  Honestly, I can't find a postiive to this, as it's not even close to being a 
# #  compute bottleneck.
# def batched_weighted_sum(weights, vecs, vecs2, vec3, netSize, batch_size=500):
#     ''' Takes weighted sum of the vectors in vec by breaking them into groups of batch_size, dot 
#     products them together along the first dimension and suming the results across all batches

#     TODO: I don't understand how this code is doing anything other than just doting weights with vecs 
#     and summing... I've run this function a couple of times and it doesn't seem to depend on batch_size
#     at all.

#     Parameters
#     ---------- 
#     weights: list
#         List of numeric weights
#     vecs: list
#         List of numeric vectors the same shape as weights
#     batch_size: int
#         Size of expected batches

#     Returns
#     ------- 
#     total: np.Array
#         Weighted sum of vectors
#     num_items_summed: int
#         Length of weights
#     '''
#     import time


#     t1s = time.time()

#     total = 0.
#     num_items_summed = 0
#     for batch_weights, batch_vecs in zip(itergroups(weights, batch_size), itergroups(vecs, batch_size)):
#         assert len(batch_weights) == len(batch_vecs) <= batch_size
        
#         # print("inside group loop")
#         # # print(batch_weights)
#         # # print(batch_vecs)

#         # print("\n")
#         # print(f"Length of weights: {len(batch_weights)}")
#         # print(f"Dims of vecs: {len(batch_vecs), len(batch_vecs[0])}")
#         # print(np.asarray(batch_weights).shape)
#         # print(np.asarray(batch_vecs).shape)
#         # print("\n")

        
#         total += np.dot(np.asarray(batch_weights, dtype=np.float32),
#                         np.asarray(batch_vecs, dtype=np.float32))
#         num_items_summed += len(batch_weights)


#     t1e = time.time()

#     # print(weights)
#     # print("length of weights: ", len(weights))
#     # print(vecs)
#     # print(vecs2)

#     # print(type(weights))
#     # print(weights.dims)
#     # print(type(vec))
#     # print(vec.dims)

#     t2s = time.time()

#     total2 = 0.
#     num_items_summed2 = 0


#     for vec in vecs2:
#         total2 += weights[num_items_summed2] * np.asarray(vec, dtype=np.float32)
#         num_items_summed2 += 1

#     t2e = time.time()

#     t3s = time.time()

#     total3 = np.zeros(netSize, dtype=np.float32)
#     num_items_summed3 = 0
#     for vec in vec3:
#         total3 += (weights[num_items_summed3] * vec)
#         num_items_summed3 += 1

#     t3e = time.time()

#     print(f"Total is {total} with length {len(total)}")
#     print(f"Total2 is {total2} with length {len(total2)}")
#     print(f"Total3 is {total3} with length {len(total3)}")
#     print(f"Type of 1 is: {type(total)}")
#     print(f"Type of 2 is: {type(total2)}")
#     print(f"Type of 3 is: {type(total3)}")
#     print(num_items_summed)
#     print(num_items_summed2)
#     print(num_items_summed3)
#     print(f"Time for method 1 is: {t1e - t1s}")
#     print(f"Time for method 2 is: {t2e - t2s}")
#     print(f"Time for method 3 is: {t3e - t3s}")




#     if((abs(total - total2) < 1e-4).all()):
#         print("\n\nTotals 1 & 2 in 'batch_weighted_sum()' are the same")

#     if((abs(total - total3) < 1e-4).all()):
#         print("Totals 1 & 3 in 'batch_weighted_sum()' are the same\n\n")


#     return total, num_items_summed
