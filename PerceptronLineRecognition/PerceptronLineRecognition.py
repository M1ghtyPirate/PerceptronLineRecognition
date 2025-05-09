from math import ceil
import os
from random import random
import re
from typing import Callable
from PIL import Image

def getFiles(dir: str, pattern: str) -> list[str]:
    '''
    Формирования списка путей к файлам

    :param dir: путь к папке
    :param pattern: regex имени файлов

    :return пути к файлам
    '''
    return [f'{dir}\{f}' for f in os.listdir(dir) if os.path.isfile(f'{dir}\{f}') and re.fullmatch(pattern, f)]

def getImagePattern(img: Image.Image) -> list[int]:
    '''
    Формирование образа изображения

    :param img: однобитное изображение

    :return образ (значения пикселей)
    '''
    imgPattern = []
    for j in range(img.height):
        for i in range(img.width):
            imgPattern.append(img.getpixel((i, j)))
    return imgPattern

def getTrainingSet(imageFiles: list[str]) -> tuple[list[tuple[list[int], bool]], tuple[str, str]]:
    '''
    Формирование сета для обучения
    
    :param imageFiles: пути к изображениям обучающего сета

    :return ([(<образ>, <активация происходит>)], (<результат активации>, <результат неактивации>))
    '''
    images = list(map(lambda x: Image.open(x), imageFiles))
    if max(img.width for img in images) != min(img.width for img in images) or max(img.height for img in images) != min(img.height for img in images):
        raise ValueError("Переданы изображения с различным разрешением.")
    names = []
    for imageFile in imageFiles:
        name = imageFile.split("\\")[-1].split("_")[0]
        if name in names:
            continue
        names.append(name)
    if len(names) != 2:
        raise ValueError("Переданы изображения не с двумя вариантами распознавания.")
    trainingSet = []
    for i in range(len(images)):
        img = images[i]
        imgPattern = getImagePattern(img)
        name = imageFiles[i].split("\\")[-1].split("_")[0]
        activation = names[0] == name
        trainingSet.append((imgPattern, activation))
    return (trainingSet, (names[0], names[1]))

def getPerceptron(trainingSet: tuple[list[tuple[list[int], bool]], tuple[str, str]], learningRate: float = 0.01, activationFunction: Callable[[float], bool] = lambda v: v >= 0, maxCycles: int = 0) -> tuple[list[int], float, Callable[[float], bool], tuple[str, str]]:
    '''
    Формирование весов и смещения перцептрона

    :param trainingSet: сет для обучения
    :param learningRate: множитель корректировки весов и смещения
    :param activationFunction: функция активации
    :param maxCycles: ограничение по циклам обучения

    :return ([<вес>], <смещение>, <функция активации>, (<результат активации>, <результат неактивации>))
    '''
    if max(len(p[0]) for p in trainingSet[0]) != min(len(p[0]) for p in trainingSet[0]) or learningRate < 0 or learningRate > 1 or maxCycles < 0:
        raise ValueError("Переданы некорректные аргументы для формирования весов и смещения.")
    print(f'Формирование весов и смещения' + ('.' if maxCycles == 0 else f' с ограничением в <{maxCycles}> циклов.'))
    patternCount = len(trainingSet[0])
    patternLength = len(trainingSet[0][0][0])
    weights = [0 for i in range(patternLength)]
    #weights = [random() * 2 - 1 for i in range(patternLength)]
    bias = 0
    #bias = random() - 0.5
    maxCycles = maxCycles if maxCycles > 0 else -1
    currentCycle = 0
    reportPercent = 0
    while maxCycles < 0 or currentCycle < maxCycles:
        weightsCorrected = False
        for i in range(patternCount):
            pattern = trainingSet[0][i]
            s = sum(list(map(lambda j: pattern[0][j] * weights[j], range(patternLength)))) + bias
            y = activationFunction(s)
            t = pattern[1]
            dy = (t - y) * learningRate
            for j in range(patternLength):
                dw = pattern[0][j] * dy
                weights[j] += dw
                weightsCorrected = weightsCorrected or abs(dw) > 0
            bias += dy
            weightsCorrected = weightsCorrected or abs(dy) > 0
        if not weightsCorrected:
            break
        currentCycle += 1
        if maxCycles > 0:
            percentDone = ceil(100 * currentCycle / maxCycles)
            if percentDone >= reportPercent:
                print(f'{reportPercent}% максимального количества циклов завершено.')
                reportPercent += 10
    print(f'Завершено за {currentCycle} циклов.')
    return (weights, bias, activationFunction, trainingSet[1])

def recognizePattern(pattern: list[int], perceptron: tuple[list[int], float, Callable[[float], bool], tuple[str, str]]) -> tuple[bool, str]:
    '''
    Распознавание образа

    :param pattern: образ
    :param perceptron: данные обученного перцептрона

    :return (<активация произошла>, <результат>)
    '''
    if len(perceptron[0]) != len(pattern):
        raise ValueError("Переданы некорректные данные весов перцептрона.")
    patternLength = len(pattern)
    weights = perceptron[0]
    bias = perceptron[1]
    activationFunction = perceptron[2]
    names = perceptron[3]
    s = sum(list(map(lambda i: pattern[i] * weights[i], range(patternLength)))) + bias
    y = activationFunction(s)
    result = (y, names[0] if y else names[1])
    print(f'Возбуждение перцептрона: <{s}>')
    # print(f'Результат распознавания: <{result}>')
    return result

trainingImageSet = getFiles("TrainingSet", "^.*\.bmp$")
#trainingImageSet = getFiles("RecognitionSet", "^.*\.bmp$")
recognitionImageSet = getFiles("RecognitionSet", "^.*\.bmp$")
#recognitionImageSet = getFiles("TrainingSet", "^.*\.bmp$")
print(f'Сет обучения:')
trainingSet = getTrainingSet(trainingImageSet)
for i in range(len(trainingImageSet)):
    print()
    print(trainingImageSet[i])
    print(f'{trainingSet[0][i]} - {trainingSet[1][0] if trainingSet[0][i][1] else trainingSet[1][1]}')
perceptron = getPerceptron(trainingSet, maxCycles = 30000)
print()
print(f'Веса перцептрона со смещением <{perceptron[1]}> и результатами активации/неактивации <{perceptron[3][0]}/{perceptron[3][1]}>:')
print(perceptron[0])
print()
print(f'Сет распознавания:')
correctPredictions = 0
for file in recognitionImageSet:
    print()
    print(file)
    recognitionResult = recognizePattern(getImagePattern(Image.open(file)), perceptron)
    print(f'Результат распознавания: <{recognitionResult}>')
    if recognitionResult[1] == file.split("\\")[-1].split("_")[0]:
        correctPredictions += 1
print(f'Точность распознавания: {correctPredictions}/{len(recognitionImageSet)} - {round(correctPredictions / len(recognitionImageSet), 2) * 100}%')