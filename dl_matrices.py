# from query import search

# result = search(nzbounds=(1000000,1000000000), isspd=True, limit=10000000000, dtype='real')
# result.download(extract=True,destpath="./mm/")


from ssgetpy import search, fetch
import sys
import os


download = 1
def dl_save_list(matrix_directory, matrix_list_path):
    # specify what matrices should be downloaded

    result = search(nzbounds=(100000, 300000), isspd=True, limit=10000000000, dtype='real')
    # uncomment the following line to download all SPD matrices
    # result = search(nzbounds=(100000, 150000000), isspd=True, limit=1000000000000000, dtype='real')

    if download == 1:
        result.download(extract=True, destpath=matrix_directory)
    # generate the list of downloaded matrices
    matrix_list = []
    for matrix in result:
        # join two paths to get the full path of the matrix and append it to the matrix list
        if matrix.rows == matrix.cols:
            matrix_list.append(os.path.join(matrix.name, matrix.name + '.mtx'))
    # write the matrix list to a txt file in the matrix_list directory
    with open(matrix_list_path, 'w') as f:
        for item in matrix_list:
            f.write("%s\n" % item)


if __name__ == '__main__':
    dl_save_list(sys.argv[1], sys.argv[2])
