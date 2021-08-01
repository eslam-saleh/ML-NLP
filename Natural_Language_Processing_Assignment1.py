from collections import defaultdict
from tkinter import *

# (word) -> count
unigram = defaultdict(int)
# (word1, word2) -> count
bigram = defaultdict(lambda: defaultdict(int))
# (word1, word2, word3) -> count
trigram = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))


def read_corpus():
    for i in range(1, 910):
        f = open(f'corpus/Khaleej-2004-Economy/economy-{i}.html', encoding='utf-8')
        calculate_count(f.read())
        f.close()


def calculate_count(doc):
    global unigram, bigram, trigram
    doc = re.findall('[^ \.\*\s\t\n\'\"\(\)\{\}\[\]\\<>!@#$%&?,:;،؟؛]+', doc)

    # increasing count for each unique word / sequence of words
    for i in range(len(doc) - 2):
        unigram[doc[i]] = unigram[doc[i]] + 1
        bigram[doc[i]][doc[i + 1]] = bigram[doc[i]][doc[i + 1]] + 1
        trigram[doc[i]][doc[i + 1]][doc[i + 2]] = trigram[doc[i]][doc[i + 1]][doc[i + 2]] + 1

    # increasing count for unhandled last 2 indices in the prev loop
    unigram[doc[-2]] = unigram[doc[-2]] + 1
    unigram[doc[-1]] = unigram[doc[-1]] + 1
    bigram[doc[-2]][doc[-1]] = bigram[doc[-2]][doc[-1]] + 1


def calc_prob(words):
    global unigram, bigram, trigram
    prob = 0.0
    if len(words) == 2:
        # p(y|x) = count(x, y) / count(x)
        prob = bigram[words[0]][words[1]] / unigram[words[0]]
    elif len(words) == 3:
        # p(z|x, y) = count(x, y, z) / count(x, y)
        prob = trigram[words[0]][words[1]][words[2]] / bigram[words[0]][words[1]]
    return prob


def predict_word(sentence):
    # return if text box is empty
    if sentence == '':
        return None

    global unigram, bigram, trigram
    words = re.split(' ', sentence)
    # make predictions using bigram if text box contains only one word
    if len(words) == 1:
        if bigram.get(words[0]) is not None:
            predictions = []
            for key, val in bigram[words[0]].items():
                predictions.append([key, calc_prob([words[0], key])])
            # sort prediction based on highest prob first
            predictions.sort(key=lambda x: x[1], reverse=True)
            if len(predictions) > 10:
                return predictions[:11]
            else:
                return predictions
    # make predictions using trigram if text box contains two words
    # and if it contains more than two words it predicts using the last two words
    elif len(words) > 1:
        if trigram.get(words[-2]) is not None:
            if trigram.get(words[-2]).get(words[-1]) is not None:
                predictions = []
                for key, val in trigram[words[-2]][words[-1]].items():
                    predictions.append([
                        key,
                        calc_prob([words[-2], words[-1], key])
                    ])
                # sort prediction based on highest prob first
                predictions.sort(key=lambda x: x[1], reverse=True)
                if len(predictions) > 10:
                    return predictions[:11]
                else:
                    return predictions


def list_predictions(sentence, predictions):
    lst = []
    # each prediction contains [next_word, probability]
    if predictions is not None:
        for p in predictions:
            # only appends th next_word of each prediction based on highest prob first
            lst.append((sentence, p[0]))
    else:
        return None
    return lst


#   ================================================


def check_key(event):
    # get value from text field
    value = e.get().strip()
    # get list of predictions
    result = list_predictions(value, predict_word(value))
    # update data in listbox
    update_view(result)


def update_view(data):
    global root, lb
    # clear previous data
    lb.delete(0, END)
    # put new data
    if data is not None and data != []:
        switch_lb(True)
        for item in data:
            tmp = ' '.join(item)
            lb.insert(END, tmp)
    else:
        switch_lb(False)


def switch_lb(visible):
    global root, lb
    # lb.destroy()
    if visible:
        lb.configure(width=60, bg='#FFFFFF', bd='1')
        # lb = Listbox(root, width=60, font=("Arial", 16), justify=RIGHT)
        # handle item selection event
        # lb.unbind((lb, root, "all"))
        lb.bind('<<ListboxSelect>>', select_item)
    else:
        lb.configure(width=0, bg='#F0F0F0', bd='0')
        # hide lb using same bg color
        # lb = Listbox(root, bg='#F0F0F0', bd='0')
        # redirect any clicks to the root window
        # lb.bind('<<ListboxSelect>>', None)
        # lb.bindtags((lb, root, "all"))
    # lb.pack()


def select_item(event):
    selection = event.widget.curselection()
    if selection:
        # remove current text box value
        e.delete(0, END)
        # add current listbox selected value to the text box
        e.insert(0, event.widget.get(selection[0]))
        # continue prediction
        check_key(None)


if __name__ == '__main__':
    read_corpus()
    print('Distinct Words =', sum(unigram.values()), 'words')

    # gui window
    root = Tk()
    root.geometry('800x420')
    root.title('Auto Fill')

    label = Label(root, height=2, text="Google", font=("Arial", 32)).pack()

    # creating text box
    e = Entry(root, width=60, font=("Arial", 16), justify=RIGHT)
    e.focus_set()
    e.bind('<KeyRelease>', check_key)
    e.pack()

    # creating list box
    lb = Listbox(root, width=0, font=("Arial", 16), justify=RIGHT, bg='#F0F0F0', bd='0')
    # lb.bindtags((lb, root, "all"))
    lb.pack()

    # deploy gui window
    root.mainloop()
