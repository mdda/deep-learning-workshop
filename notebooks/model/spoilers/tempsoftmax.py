def temperature_softmax(p, T):
    z = np.log(p)
    z /= T
    return np.exp(z) / np.exp(z).sum()

primer = np.random.choice(primers) + '\n'
print('PRIMER: ' + primer)

for T in [1.5, 1, 0.5, 0.1, 0.05]:
    sentence = ''
    hid = np.zeros((1, RNN_HIDDEN_SIZE), dtype='float32')
    hid2 = np.zeros((1, RNN_HIDDEN_SIZE), dtype='float32')
    x = np.zeros((1, 1, VOCAB_SIZE), dtype='float32')

    for c in primer:
        p, hid, hid2 = predict_fn(x, hid, hid2)
        x[0, 0, :] = CHAR_TO_ONEHOT[c]
    
    for _ in range(500):
        p, hid, hid2 = predict_fn(x, hid, hid2)
        p = temperature_softmax(p, T)
        p = p/(1 + 1e-6)
        s = np.random.multinomial(1, p)
        sentence += IX_TO_CHAR[s.argmax(-1)]
        x[0, 0, :] = s
        if sentence[-1] == '\n':
            break
        
    print('GENERATED (Temperature = {}): {}\n'.format(T, sentence))
