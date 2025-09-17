#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["requests"]
# ///

"""
    README:

    Cordle is Curses based Wordle so you can
    play in your terminal!

    You will need to enable color support
    Examples:
    fish: set TERM xterm-256color
    bash: export TERM=xterm-256color

    You will need to install uv next

    Lastly chmod +x this file, then you will be able
    to execute it by running ./cordle.py, uv should
    take care of the rest
    
"""

from curses import wrapper
import curses
import random
import sys
import os
import requests

words = {}

def loadText(path):
    res = ""
    with open(path, "r") as f:
        res = f.read()
    return res

def writeText(path, text):
    with open(path, "w") as f:
        f.write(text)

def sort(path):
    filePrefix = path.split(".")[0]
    words = [w for w in loadText(path).split("\n") if "'" not in w]
    wordsByLen = {}
    for word in words:
        lenWord = str(len(word)).zfill(2)
        if lenWord not in wordsByLen.keys():
            wordsByLen[lenWord] = []
        wordsByLen[lenWord].append(word)
    for lenWords in wordsByLen.keys():
        writeText(f"{filePrefix}_{lenWords}.txt", "\n".join(wordsByLen[lenWords]))

def checkWordExists(word):
    lenWord = str(len(word)).zfill(2)
    if word in words[lenWord]:
        return True
    else:
        return False

def randomWord(size, language):
    lenWord = str(size).zfill(2)
    if lenWord not in words.keys():
        path = f"words/{language}_{lenWord}.txt"
        words[lenWord] = loadText(path).split("\n")
    return random.choice(words[lenWord])


def fillBg(stdscr, colorPair):
    for y in range(36):
        stdscr.addstr(y, 0, "â–ˆâ–ˆ"*22, curses.color_pair(colorPair))
    
def checkGuess(word, guess):
    success = True
    res = []
    for it, char in enumerate(guess):
        cres = 0
        if word[it] == char:
            cres = 2
        elif char in list(word):
            success = False
            cres = 5
        else:
            success = False
        res.append(cres)
    return res, success

def downloadFile(url, path):
    r = requests.get(url, stream=True)

    with open(path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=16*1024):
            f.write(chunk)

def init():
    if not os.path.exists("words"):
        os.makedirs("words")
    if not os.path.exists("words/english.txt"):
        downloadFile("https://raw.githubusercontent.com/kkrypt0nn/wordlists/refs/heads/main/wordlists/languages/english.txt", "words/english.txt")
    if not os.path.exists("words/english_01.txt"):
        sort("words/english.txt")

alphabet = list("abcdefghijklmnopqrstuvwxyz")

def main(stdscr):
    success = False
    curses.start_color()
    bg = [1000, 1000, 1000]
    boxColor = [0, 1000, 0]
    red = [1000, 0, 0]
    orange = [1000, 500, 0]
    wordSize = 5
    if len(sys.argv) > 1:
        wordSize = int(sys.argv[1])
    word = randomWord(wordSize, "english")
    guess = []
    guesses = []
    guessesColors = []
    curses.init_color(1, bg[0], bg[1], bg[2])
    curses.init_color(2, boxColor[0], boxColor[1], boxColor[2])
    curses.init_color(3, red[0], red[1], red[2])
    curses.init_color(4, orange[0], orange[1], orange[2])
    curses.init_pair(1, 1, 1)
    curses.init_pair(2, 1, 2)
    curses.init_pair(3, 2, 2)
    curses.init_pair(4, 0, 3)
    curses.init_pair(5, 1, 4)
    # Clear screen
    stdscr.clear()
    fillBg(stdscr, 1)
    stdscr.addstr(1, 1, "â–ˆ", curses.color_pair(3))
    stdscr.addstr(1, 2, "CORDLE", curses.color_pair(2))
    stdscr.addstr(1, 8, "â–ˆ", curses.color_pair(3))
    spacing = 4
    while True:
        # stdscr.clear()
        # fillBg(stdscr, 1)
        for it, tguess in enumerate(guesses):
            stdscr.addstr(4+spacing*it, 1, "â–ˆ"*(6+len(word)*2), curses.color_pair(3))
            stdscr.addstr(4+spacing*it, 1, "â–ˆâ–ˆâ–ˆ", curses.color_pair(3))
            for iit, char in enumerate(tguess):
                curColor = guessesColors[it][iit]
                if iit == 0:
                    stdscr.addstr(4+spacing*it, 4+iit*2, char, curses.color_pair(curColor))
                else:
                    stdscr.addstr(4+spacing*it, 4+iit*2-1, " "+ char, curses.color_pair(curColor))
            stdscr.addstr(4+spacing*it, 4+len(word)*2, "â–ˆâ–ˆâ–ˆ", curses.color_pair(3))
            stdscr.addstr(5+spacing*it, 1, "â–ˆ"*(6+len(word)*2), curses.color_pair(3))
        if (len(guesses) < 6) and (not success):
            stdscr.addstr(3+spacing*len(guesses), 1, "â–ˆ"*(6+len(word)*2), curses.color_pair(3))
            stdscr.addstr(4+spacing*len(guesses), 1, "â–ˆâ–ˆâ–ˆ", curses.color_pair(3))
            guessPad = list(" "*len(word))
            for it, char in enumerate(guess):
                guessPad[it] = char
            stdscr.addstr(4+spacing*len(guesses), 4, " ".join(guessPad), curses.color_pair(2))
            stdscr.addstr(4+spacing*len(guesses), 3+len(word)*2, "â–ˆâ–ˆâ–ˆâ–ˆ", curses.color_pair(3))
            stdscr.addstr(5+spacing*len(guesses), 1, "â–ˆ"*(6+len(word)*2), curses.color_pair(3))
        else:
            if success:
                stdscr.addstr(33, 1, "Success", curses.color_pair(2))
            else:
                stdscr.addstr(33, 1, "Failed", curses.color_pair(4))
        stdscr.addstr(28, 1, "Backspace to erase characters", curses.color_pair(2))
        stdscr.addstr(29, 1, "Enter to submit answer", curses.color_pair(2))
        stdscr.addstr(30, 1, "Comma to restart", curses.color_pair(2))
        stdscr.addstr(31, 1, "Period to end program", curses.color_pair(2))
        stdscr.refresh()
        event = stdscr.getkey()
        stdscr.addstr(35, 1, " "*20, curses.color_pair(1))
        if len(event) == 1:
            if event.lower() in alphabet:
                if len(guess) < len(word):
                    guess.append(event)
            else:
                match event:
                    case "\n":
                        if (len(guess) == len(word)) and (len(guesses) < 6):
                            if checkWordExists("".join(guess)):
                                cguess = [c for c in guess]
                                guesses.append(cguess)
                                checkedGuess, success = checkGuess(word, cguess)
                                guessesColors.append(checkedGuess)
                                guess = []
                            else:
                                stdscr.addstr(35, 1, " "*20, curses.color_pair(1))
                                stdscr.addstr(35, 1, "Word does not exist", curses.color_pair(4))
                        elif (len(guess) < len(word)):
                            stdscr.addstr(35, 1, " "*20, curses.color_pair(1))
                            stdscr.addstr(35, 1, "Guess is too short", curses.color_pair(4))
                    case ".":
                        break
                    case "\x7f":
                        if len(guess) > 0:
                            del guess[-1]
                    case ",":
                        word = randomWord(len(word), "english")
                        success = False
                        guess = []
                        guesses = []
                        guessesColors = []
                        stdscr.clear()
                        fillBg(stdscr, 1)
                        stdscr.addstr(1, 1, "â–ˆ", curses.color_pair(3))
                        stdscr.addstr(1, 2, "CORDLE", curses.color_pair(2))
                        stdscr.addstr(1, 8, "â–ˆ", curses.color_pair(3))

if __name__ == "__main__":
    print("Initializing")
    init()
    print("Done initializing")
    wrapper(main)
