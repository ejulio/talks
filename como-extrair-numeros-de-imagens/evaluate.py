for (i, batch) in enumerate(val_dataloader):
    (image, input_labels, output_labels) = batch
    image = Variable(image.cuda())
    input_labels = Variable(input_labels.cuda())

    decoder.reset_state(batch_size=1)
    image_embedding = encoder(image)
    decoder.input_images(image_embedding)

    digit = input_labels[0, 0]
    sequence = []
    while len(sequence) < svhn.N_LABELS and digit.data[0] != svhn.STOP:
        prediction = decoder(digit)
        digit = torch.max(prediction, dim=1)[1]
        sequence.append(digit.data[0])

    while len(sequence) < svhn.N_LABELS:
        sequence.append(svhn.PAD)
