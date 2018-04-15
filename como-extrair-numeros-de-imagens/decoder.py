class Decoder(nn.Module):

    def __init__(self, n_digits):
        super(Decoder, self).__init__()
        self._n_digits = n_digits
        self.digit_embeddings = nn.Embedding(n_digits, EMBEDDING_SIZE)
        init.uniform(self.digit_embeddings.weight, -1, 1)

        self.lstm = nn.LSTM(EMBEDDING_SIZE, STATE_SIZE, batch_first=True)
        init.normal(self.lstm.weight_ih_l0, 0, 0.01)
        init.constant(self.lstm.bias_ih_l0, 0)
        init.normal(self.lstm.weight_hh_l0, 0, 0.01)
        init.constant(self.lstm.bias_hh_l0, 0)

        self.fc2digit = nn.Linear(STATE_SIZE, n_digits)
        init.normal(self.fc2digit.weight, 0, 0.01)
        init.constant(self.fc2digit.bias, 0)

    def reset_state(self, batch_size):
        self._state = (Variable(torch.zeros(1, batch_size,
                                            STATE_SIZE).cuda()),
                       Variable(torch.zeros(1, batch_size,
                                            STATE_SIZE).cuda()))

    def input_images(self, images_embeddings):
        images_embeddings = images_embeddings.view(-1, 1, EMBEDDING_SIZE)
        (_, self._state) = self.lstm(images_embeddings, self._state)

    def forward(self, digits):
        digits_embeddings = self.digit_embeddings(digits)
        if not self.training:
            digits_embeddings = digits_embeddings.view(1, 1, EMBEDDING_SIZE)

        (out, self._state) = self.lstm(digits_embeddings, self._state)

        out = out.contiguous().view(-1, STATE_SIZE)
        out = self.fc2digit(out)
        predictions = F.log_softmax(out)
        return predictions