class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = self._conv(3, 16)
        self.conv2 = self._conv(16, 16)
        self.conv3 = self._conv(16, 32)
        self.conv4 = self._conv(32, 32)
        self.conv5 = self._conv(32, 64)
        self.conv6 = self._conv(64, 64)
        self.conv7 = self._conv(64, 128)
        self.conv8 = self._conv(128, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = self._fc(128 * 6 * 6, 2048)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = self._fc(2048, 1024)
        self.drop2 = nn.Dropout(p=0.2)
        self.fc3 = self._fc(1024, EMBEDDING_SIZE)

    def _conv(self, input_channels, output_channels):
        conv = nn.Conv2d(input_channels, output_channels,
                         kernel_size=3, padding=1)
        init.kaiming_normal(conv.weight, 0)
        init.constant(conv.bias, 0)
        return conv

    def _fc(self, input_units, output_units):
        fc = nn.Linear(input_units, output_units)
        init.normal(fc.weight, 0, 0.01)
        init.constant(fc.bias, 0)
        return fc

    def forward(self, images):
        out = F.relu(self.conv1(images))
        out = F.relu(self.conv2(out))
        out = self.pool(out)
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.pool(out)
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        out = self.pool(out)
        out = F.relu(self.conv7(out))
        out = F.relu(self.conv8(out))
        out = self.pool(out)
        out = out.view(-1, 6 * 6 * 128)
        out = self.drop1(out)
        out = F.relu(self.fc1(out))
        out = self.drop2(out)
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        return out