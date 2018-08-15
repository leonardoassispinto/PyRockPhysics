# -*- coding: utf-8 -*-

import numpy as np
import os
import hashlib

infinite_char = '\xe2\u02c6\u017e'
infinite_char_replacement = '9999999'

def hash_file(file_name):
    hash_md5 = hashlib.md5()
    chuch_size = 4096
    with open(file_name, "rb") as f:
        for chunk in iter(lambda: f.read(chuch_size), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_unique_buffer_filename(filename):
    return "{}.npz".format(hash_file(filename)[:16])

def load_psdata_bufferized(filename, deletefile, bufferdirectory="."):
    bufferfilename = get_unique_buffer_filename(filename)
    bufferfilepath = os.path.join(bufferdirectory, bufferfilename)

    if os.path.exists(bufferfilepath):
        npfile = np.load(bufferfilepath)
        return npfile["time"], npfile["waves"]

    time, waves = load_psdata(filename, deletefile)

    if not os.path.exists(bufferdirectory):
        os.makedirs(bufferdirectory)

    if os.path.isdir(bufferdirectory):
        np.savez(bufferfilepath, time=time, waves=waves)

    return time, waves

def load_psdata(filename, deletefile=True):
    # nesse caso o nome do arquivo é inválido
    if not os.path.exists(filename):
        print("File {} does not exist".format(filename))
        return None

    res = convert_pico_file(filename, "txt")

    # nesse caso a conversão falhou
    if not res:
        print("Erro ao converter o arquivo {}".format(filename))
        return None
    
    txtfilename = filename.replace(".psdata", ".txt")

    # numpy está bugado
    # with open(txtfilename, 'r', encoding='utf-8') as f:
        # data = np.loadtxt(f, delimiter=',', skiprows=3)
    
    data = open_csv(txtfilename, delimiter="\t", skiprows=3)

    # deleta o arquivo caso desejado
    if deletefile:
        command = 'del {}'.format(txtfilename)
        res = os.system(command)
        if res != 0:
            print("Erro ao deletar o arquivo temporário {}".format(txtfilename))
    
    time = data[:, 0].flatten() # transforma matriz de 1 coluna em vetor
    waves = data[:, 1:].transpose()

    return time, waves

def load_csv(filename, delimiter=',', skiprows=0):
    data = open_csv(filename, delimiter=delimiter, skiprows=skiprows)

    time = data[:, 0].flatten() # transforma matriz de 1 coluna em vetor
    waves = data[:, 1:].transpose()

    return time, waves

def open_csv(filename, delimiter=',', skiprows=0):
    with open(filename, 'r') as f:
        for i in range(skiprows):
            f.readline()
        
        data = []
        linenumber = skiprows
        for line in f:
            linenumber += 1
            if not line.strip():
                continue
            if infinite_char in line:
                line = line.replace(infinite_char, infinite_char_replacement)
            data.append(list(map(float, line.split(delimiter))))
        
    data = np.array(data)

    data[data == float(infinite_char_replacement)] = np.nan
    data[data == -float(infinite_char_replacement)] = np.nan

    return np.array(data)

def convert_pico_file(filename, fmt='csv'):
    command = 'picoscope /c "{}" /f {} /q /b all'.format(filename, fmt)
    res = os.system(command)
    # se retornar 0 significa que rodou OK
    return res == 0

def fix_filename(filename):
    if filename.endswith(".psdata"):
        return filename
    elif "." not in filename:
        return filename + ".psdata"
    else:
        # nesse caso o nome do arquivo é inválido
        return None
   
