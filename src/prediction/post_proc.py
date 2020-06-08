import copy

def delete_word(selected_text, words, before=True, after=True, n_word_lim=None, slc_noteq_txt=False, text=None):
    """
    If the first or last word in selected_text is one of the words, it is removed.
    """
    slc_txt = str(copy.copy(selected_text))
    splt_slc_text = slc_txt.split()
    
    if len(slc_txt) == 0:
        return ""

    if slc_noteq_txt:
        tmp_txt = text if text[0] != " " else text[1:]
        tmp_slc_txt = selected_text if selected_text[0] != " " else selected_text[1:]
        if tmp_txt == tmp_slc_txt:
            return slc_txt

    if n_word_lim is None or len(splt_slc_text) <= n_word_lim:
        if before:
            if len(splt_slc_text) > 0:
                for w in words:
                    if w == splt_slc_text[0]:
                        splt_slc_text = splt_slc_text[1:]
                        slc_txt = ' '.join(splt_slc_text)
                        break
        if after:
            if len(splt_slc_text) > 0:
                for w in words:
                    if w == splt_slc_text[-1]:
                        splt_slc_text = splt_slc_text[:-1]
                        slc_txt = ' '.join(splt_slc_text)
                        break
        
    return slc_txt

def add_word(text, selected_text, words, before=True, after=True, n_word_lim=None):
    """
    If the word before or after the selected_text is one of the words, it is added.
    """
    slc_txt = str(copy.copy(selected_text))
    txt = str(copy.copy(text))
    before_idx = txt.find(slc_txt) - 1
    after_idx = before_idx + 1 + len(slc_txt)
    
    splt_before_text = txt[:before_idx+1].split()
    splt_after_text = txt[after_idx:].split()
    
    if n_word_lim is None or len(slc_txt.split()) <= n_word_lim:
        if before:
            if len(splt_before_text) > 0:
                for w in words:
                    if w == splt_before_text[-1]:
                        slc_txt = splt_before_text[-1] + ' ' + slc_txt
                        break
        if after:
            if len(splt_after_text) > 0:
                for w in words:
                    if w == splt_after_text[0]:
                        slc_txt = slc_txt + ' ' + splt_after_text[0]
                        break
        
    return slc_txt

def add_char_recursive(text, selected_text, chars, before=True, after=True, n_word_lim=None):
    """
    If the char before or after the selected_text is one of the chars, it is added recursively.

    Args:
        chars : list of single char
    """
    slc_text = str(copy.copy(selected_text))
    txt = str(copy.copy(text))
    before_idx = txt.find(slc_text) - 1
    after_idx = before_idx + 1 + len(slc_text)
    
    if n_word_lim is None or len(slc_text.split()) <= n_word_lim:
        if before:
            while True:
                if before_idx >= 0 and txt[before_idx] in chars:
                    slc_text = text[before_idx] + slc_text
                    before_idx -= 1
                else:
                    break
        if after:
            while True:
                if after_idx < len(txt) and txt[after_idx] in chars:
                    slc_text = slc_text + txt[after_idx]
                    after_idx += 1
                else:
                    break

    return slc_text

def add_chars(text, selected_text, chars, before=True, after=True, n_word_lim=None):
    """
    If the chars before or after the selected_text is one of the chars, it is added.
    """
    slc_txt = str(copy.copy(selected_text))
    txt = str(copy.copy(text))
    before_idx = txt.find(slc_txt) - 1
    after_idx = before_idx + 1 + len(slc_txt)
    
    before_text = txt[:before_idx+1]
    after_text = txt[after_idx:]
    
    if n_word_lim is None or len(slc_txt.split()) <= n_word_lim:
        if before:
            if len(before_text) > 0:
                for ch in chars:
                    if ch == before_text[-len(ch):] and ch[-1] != before_text[-len(ch)-1]:
                        slc_txt = ch + slc_txt
                        break
        if after:
            if len(after_text) > 0:
                for ch in chars:
                    if ch == after_text[:len(ch)] and ch[0] != after_text[len(ch)]:
                        slc_txt = slc_txt + ch
                        break
        
    return slc_txt

def postproc_selected_text_v1(text, selected_text):
    slc_txt = copy.copy(selected_text)
    if slc_txt[0] == " ":
        slc_txt = slc_txt[1:]

    # all 0.558128 -> 0.552689, -0.0045

    # 0.558128 -> 0.556333, -0.0018
    slc_txt = add_chars(text, slc_txt, ["..."], before=True, after=False, n_word_lim=None)

    # 0.558128 -> 0.558683, +0.0005
    slc_txt = add_char_recursive(text, slc_txt, ["."], before=False, after=True, n_word_lim=None)

    # 0.558128 -> 0.558128, +0.0000
    slc_txt = add_char_recursive(text, slc_txt, [","], before=False, after=True, n_word_lim=None)

    # 0.558128 -> 0.556974, -0.0012
    slc_txt = delete_word(slc_txt, ["so", "soo", "sooo", "very", "more", "most"], before=True, after=False, n_word_lim=None)

    # 0.558128 -> 0.558195, +0.00007
    slc_txt = add_word(text, slc_txt, ["super"], before=True, after=False, n_word_lim=None)

    # 0.558128 -> 0.558210, +0.0001
    slc_txt = delete_word(slc_txt, ["the", "a"], before=True, after=False, n_word_lim=None)

    # 0.558128 -> 0.558333, +0.0002
    slc_txt = delete_word(slc_txt, ["really"], before=True, after=False, n_word_lim=None)

    # 0.558128 -> 0.555279, -0.0029
    slc_txt = delete_word(slc_txt, ["i", "I", "im", "Im", "i'm", "I'm", "iam", "Iam"], before=True, after=False, n_word_lim=None)

    # 0.558128 -> 0.557930, -0.0002
    slc_txt = delete_word(slc_txt, ["that"], before=True, after=False, n_word_lim=None)

    #if slc_txt != selected_text[1:]:
    #    print(slc_txt)
    #    print(selected_text)
    #    print()

    if len(slc_txt) == 0:
        slc_txt = selected_text

    return slc_txt

def postproc_selected_text_v2(text, selected_text):
    slc_txt = copy.copy(selected_text)
    if slc_txt[0] == " ":
        slc_txt = slc_txt[1:]

    n_word_lim = 5

    # all 0.558128 -> 0.553710, -0.0044

    # 0.558128 -> 0.556422, -0.0017
    slc_txt = add_chars(text, slc_txt, ["..."], before=True, after=False, n_word_lim=n_word_lim)

    # 0.558128 -> 0.558654, +0.0005
    slc_txt = add_char_recursive(text, slc_txt, ["."], before=False, after=True, n_word_lim=n_word_lim)

    # 0.558128 -> 0.558128, +0.0000
    slc_txt = add_char_recursive(text, slc_txt, [","], before=False, after=True, n_word_lim=n_word_lim)

    # 0.558128 -> 0.557049, -0.0011
    slc_txt = delete_word(slc_txt, ["so", "soo", "sooo", "very", "more", "most"], before=True, after=False, n_word_lim=n_word_lim)

    # 0.558128 -> 0.558195, +0.00007
    slc_txt = add_word(text, slc_txt, ["super"], before=True, after=False, n_word_lim=n_word_lim)

    # 0.558128 -> 0.558261, +0.0001
    slc_txt = delete_word(slc_txt, ["the", "a"], before=True, after=False, n_word_lim=n_word_lim)

    # 0.558128 -> 0.558358, +0.0002
    slc_txt = delete_word(slc_txt, ["really"], before=True, after=False, n_word_lim=n_word_lim)

    # 0.558128 -> 0.556160, -0.0020
    slc_txt = delete_word(slc_txt, ["i", "I", "im", "Im", "i'm", "I'm", "iam", "Iam"], before=True, after=False, n_word_lim=n_word_lim)

    # 0.558128 -> 0.557851, -0.0003
    slc_txt = delete_word(slc_txt, ["that"], before=True, after=False, n_word_lim=n_word_lim)

    #if slc_txt != selected_text[1:]:
    #    print(slc_txt)
    #    print(selected_text)
    #    print()

    if len(slc_txt) == 0:
        slc_txt = selected_text

    return slc_txt

def postproc_selected_text_v3(text, selected_text):
    slc_txt = copy.copy(selected_text)
    if slc_txt[0] == " ":
        slc_txt = slc_txt[1:]

    n_word_lim = 2

    # all 0.558128 -> 0.557881, -0.

    # 0.558128 -> 0.556392, -0.0018
    slc_txt = add_chars(text, slc_txt, ["..."], before=True, after=False, n_word_lim=n_word_lim)

    # 0.558128 -> 0.558552, +0.0004
    slc_txt = add_char_recursive(text, slc_txt, ["."], before=False, after=True, n_word_lim=n_word_lim)

    # 0.558128 -> 0.558126, +0.0000
    slc_txt = add_char_recursive(text, slc_txt, [","], before=False, after=True, n_word_lim=n_word_lim)

    # 0.558128 -> 0.557466, -0.0007
    slc_txt = delete_word(slc_txt, ["so", "soo", "sooo", "very", "more", "most"], before=True, after=False, n_word_lim=n_word_lim)

    # 0.558128 -> 0.558195, +0.00007
    slc_txt = add_word(text, slc_txt, ["super"], before=True, after=False, n_word_lim=n_word_lim)

    # 0.558128 -> 0.558290, +0.0001
    slc_txt = delete_word(slc_txt, ["the", "a"], before=True, after=False, n_word_lim=n_word_lim)

    # 0.558128 -> 0.558384, +0.0002
    slc_txt = delete_word(slc_txt, ["really"], before=True, after=False, n_word_lim=n_word_lim)

    # 0.558128 -> 0.559872, +0.0017
    slc_txt = delete_word(slc_txt, ["i", "I", "im", "Im", "i'm", "I'm", "iam", "Iam"], before=True, after=False, n_word_lim=n_word_lim)

    # 0.558128 -> 0.557831, -0.0003
    slc_txt = delete_word(slc_txt, ["that"], before=True, after=False, n_word_lim=n_word_lim)

    #if slc_txt != selected_text[1:]:
    #    print(slc_txt)
    #    print(selected_text)
    #    print()

    if len(slc_txt) == 0:
        slc_txt = selected_text

    return slc_txt

def postproc_selected_text_v4(text, selected_text):
    slc_txt = copy.copy(selected_text)
    if slc_txt[0] == " ":
        slc_txt = slc_txt[1:]

    n_word_lim = 3

    # all 0.558128 -> , -0.

    # 0.558128 -> 0.558655, +0.0005
    slc_txt = delete_word(slc_txt, ["i", "I", "im", "Im", "i'm", "I'm", "iam", "Iam"], before=True, after=False, n_word_lim=n_word_lim)

    #if slc_txt != selected_text[1:]:
    #    print(slc_txt)
    #    print(selected_text)
    #    print()

    if len(slc_txt) == 0:
        slc_txt = selected_text

    return slc_txt

def postproc_selected_text_v5(text, selected_text):
    slc_txt = copy.copy(selected_text)
    if slc_txt[0] == " ":
        slc_txt = slc_txt[1:]

    n_word_lim = 3

    # 0.558128 -> 0.557938, -0.0002
    #slc_txt = delete_word(slc_txt, ["very"], before=True, after=False, n_word_lim=n_word_lim)

    # 0.558128 -> 0.558128, -0.
    #slc_txt = delete_word(slc_txt, ["more"], before=True, after=False, n_word_lim=n_word_lim)

    # 0.558128 -> 0.558229, +0.0001
    slc_txt = delete_word(slc_txt, ["most"], before=True, after=False, n_word_lim=n_word_lim)

    #if slc_txt != selected_text[1:]:
    #    print(slc_txt)
    #    print(selected_text)
    #    print()

    if len(slc_txt) == 0:
        slc_txt = selected_text

    return slc_txt

def postproc_selected_text_v6(text, selected_text):
    slc_txt = copy.copy(selected_text)
    if slc_txt[0] == " ":
        slc_txt = slc_txt[1:]

    n_word_lim = 2

    # 0.558128 -> 0.557905, -0.
    #slc_txt = add_word(text, slc_txt, ["very"], before=True, after=False, n_word_lim=n_word_lim)

    # 0.558128 -> 0.558395, +0.0002
    slc_txt = add_word(text, slc_txt, ["more"], before=True, after=False, n_word_lim=n_word_lim)

    # 0.558128 -> 0.556752, -0.
    #slc_txt = add_word(text, slc_txt, ["so"], before=True, after=False, n_word_lim=n_word_lim)


    #if slc_txt != selected_text[1:]:
    #    print(slc_txt)
    #    print(selected_text)
    #    print()

    if len(slc_txt) == 0:
        slc_txt = selected_text

    return slc_txt

def postproc_selected_text_v7(text, selected_text):
    slc_txt = copy.copy(selected_text)
    if slc_txt[0] == " ":
        slc_txt = slc_txt[1:]

    # all 0.558128 -> 0.561152, +0.0030

    # 0.558128 -> 0., -0.
    #slc_txt = add_chars(text, slc_txt, ["..."], before=True, after=False, n_word_lim=n_word_lim)

    # 0.558128 -> 0., +0.
    slc_txt = add_char_recursive(text, slc_txt, ["."], before=False, after=True, n_word_lim=5)

    # 0.558128 -> 0., +0.
    slc_txt = add_char_recursive(text, slc_txt, [","], before=False, after=True, n_word_lim=5)

    # 0.558128 -> 0., -0.
    #slc_txt = delete_word(slc_txt, ["so", "soo", "sooo", "very", "more", "most"], before=True, after=False, n_word_lim=n_word_lim)

    # 0.558128 -> 0., -0.
    slc_txt = add_word(text, slc_txt, ["more"], before=True, after=False, n_word_lim=2)

    # 0.558128 -> 0., +0.
    slc_txt = add_word(text, slc_txt, ["super"], before=True, after=False, n_word_lim=5)

    # 0.558128 -> 0., +0.
    slc_txt = delete_word(slc_txt, ["the", "a"], before=True, after=False, n_word_lim=2)

    # 0.558128 -> 0., +0.
    slc_txt = delete_word(slc_txt, ["really"], before=True, after=False, n_word_lim=2)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["i", "I", "im", "Im", "i'm", "I'm", "iam", "Iam"], before=True, after=False, n_word_lim=2)

    # 0.558128 -> 0., -0.
    #slc_txt = delete_word(slc_txt, ["that"], before=True, after=False, n_word_lim=n_word_lim)

    #if slc_txt != selected_text[1:]:
    #    print(slc_txt)
    #    print(selected_text)
    #    print()

    if len(slc_txt) == 0:
        slc_txt = selected_text

    return slc_txt

def postproc_selected_text_v8(text, selected_text):
    slc_txt = copy.copy(selected_text)
    if slc_txt[0] == " ":
        slc_txt = slc_txt[1:]

    n_word_lim = 1

    # all 0.558128 -> 0., -0.

    # 0.558128 -> 0., -0.
    slc_txt = add_chars(text, slc_txt, ["..."], before=True, after=False, n_word_lim=n_word_lim)
    #slc_txt = add_char_recursive(text, slc_txt, ["."], before=True, after=False, n_word_lim=n_word_lim)

    if selected_text[0] == " ":
        comp_slc_txt = selected_text[1:]
    else:
        comp_slc_txt = selected_text
    if slc_txt != comp_slc_txt:
        print()
        print(text)
        print(slc_txt)
        print(selected_text)
        print()

    if len(slc_txt) == 0:
        slc_txt = selected_text

    return slc_txt

def postproc_selected_text_v9(text, selected_text):
    slc_txt = copy.copy(selected_text)
    if slc_txt[0] == " ":
        slc_txt = slc_txt[1:]


    # all 0.558128 -> 0., -0.

    # 0.558128 -> 0.5597
    #n_word_lim = 2
    #slc_txt = delete_word(slc_txt, ["i"], before=True, after=False, n_word_lim=n_word_lim)

    #n_word_lim = 2 # 0.558128 -> 0.558331
    #n_word_lim = 5 # 0.558128 -> 0.558191
    #slc_txt = delete_word(slc_txt, ["you"], before=True, after=False, n_word_lim=n_word_lim)

    #n_word_lim = 2 # 0.558128 -> 0.558115
    #n_word_lim = 5 # 0.558128 -> 0.557590
    #slc_txt = delete_word(slc_txt, ["it"], before=True, after=False, n_word_lim=n_word_lim)

    # 0.558128 -> 0.558290
    #n_word_lim = 2
    #slc_txt = delete_word(slc_txt, ["the"], before=True, after=False, n_word_lim=n_word_lim)

    #n_word_lim = 2 # 0.558128 -> 0.558128
    #n_word_lim = 5 # 0.558128 -> 0.558128
    #n_word_lim = None # 0.558128 -> 0.558128
    #slc_txt = delete_word(slc_txt, ["an"], before=True, after=False, n_word_lim=n_word_lim)
    
    #n_word_lim = 2 # 0.558128 -> 0.558345
    #n_word_lim = 5 # 0.558128 -> 0.558365
    #n_word_lim = None # 0.558128 -> 0.558304
    #slc_txt = delete_word(slc_txt, ["is"], before=True, after=False, n_word_lim=n_word_lim)

    #n_word_lim = 2 # 0.558128 -> 0.558128
    #n_word_lim = 5 # 0.558128 -> 0.558145
    #slc_txt = delete_word(slc_txt, ["are"], before=True, after=False, n_word_lim=n_word_lim)
    
    #n_word_lim = 2 # 0.558128 -> 0.558128
    #n_word_lim = 5 # 0.558128 -> 0.558161
    #slc_txt = delete_word(slc_txt, ["am"], before=True, after=False, n_word_lim=n_word_lim)
    
    #n_word_lim = 2 # 0.558128 -> 0.558331
    #n_word_lim = 5 # 0.558128 -> 0.558331
    #slc_txt = delete_word(slc_txt, ["be"], before=True, after=False, n_word_lim=n_word_lim)

    #n_word_lim = 2 # 0.558128 -> 0.558088
    #n_word_lim = 5 # 0.558128 -> 0.558122
    #slc_txt = delete_word(slc_txt, ["was"], before=True, after=False, n_word_lim=n_word_lim)

    #n_word_lim = 2 # 0.558128 -> 0.558026
    #n_word_lim = 5 # 0.558128 -> 0.557658
    #slc_txt = delete_word(slc_txt, ["my"], before=True, after=False, n_word_lim=n_word_lim)

    #n_word_lim = 2 # 0.558128 -> 0.558128
    #n_word_lim = 5 # 0.558128 -> 0.558161
    #slc_txt = delete_word(slc_txt, ["me"], before=True, after=False, n_word_lim=n_word_lim)

    #n_word_lim = 2 # 0.558128 -> 0.558128
    #n_word_lim = 5 # 0.558128 -> 0.558071
    #slc_txt = delete_word(slc_txt, ["and"], before=True, after=False, n_word_lim=n_word_lim)

    #n_word_lim = 2 # 0.558128 -> 0.558128
    #n_word_lim = 5 # 0.558128 -> 0.558189
    #n_word_lim = None # 0.558128 -> 0.558284
    #slc_txt = delete_word(slc_txt, ["but"], before=True, after=False, n_word_lim=n_word_lim)

    #n_word_lim = 2 # 0.558128 -> 0.558229
    #n_word_lim = 5 # 0.558128 -> 0.558229
    #slc_txt = delete_word(slc_txt, ["to"], before=True, after=False, n_word_lim=n_word_lim)

    #n_word_lim = 2 # 0.558128 -> 0.558229
    #n_word_lim = 5 # 0.558128 -> 0.558280
    #slc_txt = delete_word(slc_txt, ["just"], before=True, after=False, n_word_lim=n_word_lim)

    #n_word_lim = 2 # 0.558128 -> 0.558087
    #n_word_lim = 5 # 0.558128 -> 0.558019
    #slc_txt = delete_word(slc_txt, ["too"], before=True, after=False, n_word_lim=n_word_lim)

    #n_word_lim = 2 # 0.558128 -> 0.558181
    #n_word_lim = 5 # 0.558128 -> 0.558134
    #slc_txt = delete_word(slc_txt, ["have"], before=True, after=False, n_word_lim=n_word_lim)

    #n_word_lim = 2 # 0.558128 -> 0.558200
    #n_word_lim = 5 # 0.558128 -> 0.558149
    #slc_txt = delete_word(slc_txt, ["feel"], before=True, after=False, n_word_lim=n_word_lim)

    #n_word_lim = 2 # 0.558128 -> 0.558061
    #n_word_lim = 5 # 0.558128 -> 0.557871
    #slc_txt = delete_word(slc_txt, ["feeling"], before=True, after=False, n_word_lim=n_word_lim)


    #if selected_text[0] == " ":
    #    comp_slc_txt = selected_text[1:]
    #else:
    #    comp_slc_txt = selected_text
    #if slc_txt != comp_slc_txt:
    #    print()
    #    print(text)
    #    print(slc_txt)
    #    print(selected_text)
    #    print()

    if len(slc_txt) == 0:
        slc_txt = selected_text

    return slc_txt

def postproc_selected_text_v10(text, selected_text):
    slc_txt = copy.copy(selected_text)
    if slc_txt[0] == " ":
        slc_txt = slc_txt[1:]


    # all 0.558128 -> 0., -0.

    #n_word_lim = 5
    #slc_noteq_txt = False # 0.558128 -> 0.556191
    #slc_noteq_txt = True # 0.558128 -> 0.557165
    
    #n_word_lim = 3
    #slc_noteq_txt = False # 0.558128 -> 0.558683
    #slc_noteq_txt = True # 0.558128 -> 0.558989
    #slc_txt = delete_word(slc_txt, ["i"], before=True, after=False, n_word_lim=n_word_lim, slc_noteq_txt=slc_noteq_txt, text=text)

    n_word_lim = 2
    #slc_noteq_txt = False # 0.558128 -> 0.559760
    slc_noteq_txt = True # 0.558128 -> 0.559862
    slc_txt = delete_word(slc_txt, ["i"], before=True, after=False, n_word_lim=n_word_lim, slc_noteq_txt=slc_noteq_txt, text=text)


    #if selected_text[0] == " ":
    #    comp_slc_txt = selected_text[1:]
    #else:
    #    comp_slc_txt = selected_text
    #if slc_txt != comp_slc_txt:
    #    print()
    #    print(text)
    #    print(slc_txt)
    #    print(selected_text)
    #    print()

    if len(slc_txt) == 0:
        slc_txt = selected_text

    return slc_txt

def postproc_selected_text_v11(text, selected_text):
    slc_txt = copy.copy(selected_text)
    if slc_txt[0] == " ":
        slc_txt = slc_txt[1:]

    # all 0.558128 -> 0.562079, +0.0039

    # 0.558128 -> 0., +0.
    slc_txt = add_char_recursive(text, slc_txt, ["."], before=False, after=True, n_word_lim=5)

    # 0.558128 -> 0., +0.
    slc_txt = add_char_recursive(text, slc_txt, [","], before=False, after=True, n_word_lim=5)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["is", "are", "am", "be"], before=True, after=False, n_word_lim=5)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["i", "I", "im", "Im", "i'm", "I'm", "iam", "Iam"], before=True, after=False, n_word_lim=2)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["you"], before=True, after=False, n_word_lim=2)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["me"], before=True, after=False, n_word_lim=5)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["but"], before=True, after=False, n_word_lim=None)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["to"], before=True, after=False, n_word_lim=5)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["just"], before=True, after=False, n_word_lim=5)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["have"], before=True, after=False, n_word_lim=2)

    # 0.558128 -> 0., +0.
    slc_txt = delete_word(slc_txt, ["the", "a", "an"], before=True, after=False, n_word_lim=2)

    # 0.558128 -> 0., +0.
    slc_txt = delete_word(slc_txt, ["really"], before=True, after=False, n_word_lim=2)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["i", "I", "im", "Im", "i'm", "I'm", "iam", "Iam"], before=True, after=False, n_word_lim=2)

    # 0.558128 -> 0., +0.
    slc_txt = add_word(text, slc_txt, ["super"], before=True, after=False, n_word_lim=5)

    #if slc_txt != selected_text[1:]:
    #    print(slc_txt)
    #    print(selected_text)
    #    print()

    if len(slc_txt) == 0:
        slc_txt = selected_text

    return slc_txt

def postproc_selected_text_v12(text, selected_text):
    slc_txt = copy.copy(selected_text)
    if slc_txt[0] == " ":
        slc_txt = slc_txt[1:]

    # all 0.558128 -> 0.562079, +0.0039

    # 0.558128 -> 0., +0.
    slc_txt = add_char_recursive(text, slc_txt, ["."], before=False, after=True, n_word_lim=5)

    # 0.558128 -> 0., +0.
    slc_txt = add_char_recursive(text, slc_txt, [","], before=False, after=True, n_word_lim=5)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["but"], before=True, after=False, n_word_lim=None)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["i", "I", "im", "Im", "i'm", "I'm", "iam", "Iam"], before=True, after=False, n_word_lim=2)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["you"], before=True, after=False, n_word_lim=2)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["is", "are", "am", "be"], before=True, after=False, n_word_lim=5)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["have"], before=True, after=False, n_word_lim=2)

    # 0.558128 -> 0., +0.
    slc_txt = delete_word(slc_txt, ["really"], before=True, after=False, n_word_lim=2)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["me"], before=True, after=False, n_word_lim=5)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["to"], before=True, after=False, n_word_lim=5)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["just"], before=True, after=False, n_word_lim=5)

    # 0.558128 -> 0., +0.
    slc_txt = delete_word(slc_txt, ["the", "a", "an"], before=True, after=False, n_word_lim=2)

    # 0.558128 -> 0., +0.
    slc_txt = add_word(text, slc_txt, ["super"], before=True, after=False, n_word_lim=5)

    #if slc_txt != selected_text[1:]:
    #    print(slc_txt)
    #    print(selected_text)
    #    print()

    if len(slc_txt) == 0:
        slc_txt = selected_text

    return slc_txt

def postproc_selected_text_v13(text, selected_text):
    slc_txt = copy.copy(selected_text)
    if slc_txt[0] == " ":
        slc_txt = slc_txt[1:]

    # all 0.558128 -> 0.562398, +0.4270

    # 0.558128 -> 0., +0.
    slc_txt = add_char_recursive(text, slc_txt, ["."], before=False, after=True, n_word_lim=5)

    # 0.558128 -> 0., +0.
    slc_txt = add_char_recursive(text, slc_txt, [","], before=False, after=True, n_word_lim=5)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["but"], before=True, after=False, n_word_lim=None, slc_noteq_txt=True, text=text)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["i", "I", "im", "Im", "i'm", "I'm", "iam", "Iam"], before=True, after=False, n_word_lim=2, slc_noteq_txt=True, text=text)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["you"], before=True, after=False, n_word_lim=2, slc_noteq_txt=True, text=text)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["is", "are", "am", "be"], before=True, after=False, n_word_lim=5, slc_noteq_txt=True, text=text)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["have"], before=True, after=False, n_word_lim=2, slc_noteq_txt=True, text=text)

    # 0.558128 -> 0., +0.
    slc_txt = delete_word(slc_txt, ["really"], before=True, after=False, n_word_lim=2, slc_noteq_txt=True, text=text)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["me"], before=True, after=False, n_word_lim=5, slc_noteq_txt=True, text=text)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["to"], before=True, after=False, n_word_lim=5, slc_noteq_txt=True, text=text)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["just"], before=True, after=False, n_word_lim=5, slc_noteq_txt=True, text=text)

    # 0.558128 -> 0., +0.
    slc_txt = delete_word(slc_txt, ["the", "a", "an"], before=True, after=False, n_word_lim=2, slc_noteq_txt=True, text=text)

    # 0.558128 -> 0., +0.
    slc_txt = add_word(text, slc_txt, ["super"], before=True, after=False, n_word_lim=5)

    #if slc_txt != selected_text[1:]:
    #    print(slc_txt)
    #    print(selected_text)
    #    print()

    if len(slc_txt) == 0:
        slc_txt = selected_text

    return slc_txt

def postproc_selected_text_v14(text, selected_text):
    slc_txt = copy.copy(selected_text)
    if slc_txt[0] == " ":
        slc_txt = slc_txt[1:]

    slc_txt = slc_txt.lower()
    txt = str(text).lower()

    # all 0.558128 -> 0.562398, +0.004270

    # 0.558128 -> 0., +0.
    slc_txt = add_char_recursive(txt, slc_txt, ["."], before=False, after=True, n_word_lim=5)

    # 0.558128 -> 0., +0.
    slc_txt = add_char_recursive(txt, slc_txt, [","], before=False, after=True, n_word_lim=5)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["but"], before=True, after=False, n_word_lim=None, slc_noteq_txt=True, text=txt)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["i", "im", "i'm", "iam"], before=True, after=False, n_word_lim=2, slc_noteq_txt=True, text=txt)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["you"], before=True, after=False, n_word_lim=2, slc_noteq_txt=True, text=txt)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["is", "are", "am", "be"], before=True, after=False, n_word_lim=5, slc_noteq_txt=True, text=txt)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["have"], before=True, after=False, n_word_lim=2, slc_noteq_txt=True, text=txt)

    # 0.558128 -> 0., +0.
    slc_txt = delete_word(slc_txt, ["really"], before=True, after=False, n_word_lim=2, slc_noteq_txt=True, text=txt)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["me"], before=True, after=False, n_word_lim=5, slc_noteq_txt=True, text=txt)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["to"], before=True, after=False, n_word_lim=5, slc_noteq_txt=True, text=txt)

    # 0.558128 -> 0., -0.
    slc_txt = delete_word(slc_txt, ["just"], before=True, after=False, n_word_lim=5, slc_noteq_txt=True, text=txt)

    # 0.558128 -> 0., +0.
    slc_txt = delete_word(slc_txt, ["the", "a", "an"], before=True, after=False, n_word_lim=2, slc_noteq_txt=True, text=txt)

    # 0.558128 -> 0., +0.
    slc_txt = add_word(txt, slc_txt, ["super"], before=True, after=False, n_word_lim=5)

    #if slc_txt != selected_text[1:]:
    #    print(slc_txt)
    #    print(selected_text)
    #    print()

    if len(slc_txt) == 0:
        slc_txt = selected_text

    return slc_txt

if __name__ == '__main__':
    print('\ntest', delete_word)
    selected_text = "im kaggler in Japan"
    words = ["im", "Japan"]
    new_slc_text = delete_word(selected_text, words, before=True, after=True, n_word_lim=None)
    print(new_slc_text)
    new_slc_text = delete_word(selected_text, words, before=True, after=False, n_word_lim=None)
    print(new_slc_text)
    new_slc_text = delete_word(selected_text, words, before=False, after=True, n_word_lim=None)
    print(new_slc_text)
    new_slc_text = delete_word(selected_text, words, before=True, after=True, n_word_lim=3)
    print(new_slc_text)
    new_slc_text = delete_word(selected_text, words, before=True, after=True, n_word_lim=4)
    print(new_slc_text)

    print('\ntest', add_word)
    text = "aaa xxx im kaggler in Japan yyy bbb"
    selected_text = "im kaggler in Japan"
    words = ["xxx", "yyy"]
    new_slc_text = add_word(text, selected_text, words, before=True, after=True, n_word_lim=None)
    print(new_slc_text)
    new_slc_text = add_word(text, selected_text, words, before=True, after=False, n_word_lim=None)
    print(new_slc_text)
    new_slc_text = add_word(text, selected_text, words, before=False, after=True, n_word_lim=None)
    print(new_slc_text)
    new_slc_text = add_word(text, selected_text, words, before=True, after=True, n_word_lim=3)
    print(new_slc_text)
    new_slc_text = add_word(text, selected_text, words, before=True, after=True, n_word_lim=4)
    print(new_slc_text)

    print('\ntest', add_char_recursive)
    text = "xxx...im kaggler in Japan,,,yyy"
    selected_text = "im kaggler in Japan"
    chars = ['.', ',']
    new_slc_text = add_char_recursive(text, selected_text, chars, before=True, after=True, n_word_lim=None)
    print(new_slc_text)
    new_slc_text = add_char_recursive(text, selected_text, chars, before=True, after=False, n_word_lim=None)
    print(new_slc_text)
    new_slc_text = add_char_recursive(text, selected_text, chars, before=False, after=True, n_word_lim=None)
    print(new_slc_text)
    new_slc_text = add_char_recursive(text, selected_text, chars, before=True, after=True, n_word_lim=3)
    print(new_slc_text)
    new_slc_text = add_char_recursive(text, selected_text, chars, before=True, after=True, n_word_lim=4)
    print(new_slc_text)
    
    print('\ntest', add_chars)
    text = "aaa.x.im kaggler in Japan,y,bbb"
    selected_text = "im kaggler in Japan"
    chars = ['.x.', ',y,']
    new_slc_text = add_chars(text, selected_text, chars, before=True, after=True, n_word_lim=None)
    print(new_slc_text)
    new_slc_text = add_chars(text, selected_text, chars, before=True, after=False, n_word_lim=None)
    print(new_slc_text)
    new_slc_text = add_chars(text, selected_text, chars, before=False, after=True, n_word_lim=None)
    print(new_slc_text)
    new_slc_text = add_chars(text, selected_text, chars, before=True, after=True, n_word_lim=3)
    print(new_slc_text)
    new_slc_text = add_chars(text, selected_text, chars, before=True, after=True, n_word_lim=4)
    print(new_slc_text)
