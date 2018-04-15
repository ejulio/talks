parameters = list(encoder.parameters()) + list(decoder.parameters())
sgd = optim.SGD(parameters, lr=0.1, momentum=0.7, weight_decay=0.00005)
lr_decay_scheduler = lr_scheduler.StepLR(sgd, step_size=30, gamma=0.1)

nll = nn.NLLLoss(ignore_index=svhn.PAD)

for epoch in range(61):
    lr_decay_scheduler.step()
    print('Epoch {}, lr = {}'.format(epoch, sgd.param_groups[0]['lr']))
    decoder.train(True)
    encoder.train(True)
    losses = []
    for (i, batch) in enumerate(train_dataloader):
        (images, input_labels, output_labels) = batch
        images = Variable(images.cuda())
        input_labels = Variable(input_labels.cuda())
        output_labels = Variable(output_labels.cuda())

        sgd.zero_grad()
        batch_size = images.size()[0]
        decoder.reset_state(batch_size)
        images_embeddings = encoder(images)
        decoder.input_images(images_embeddings)
        predictions = decoder(input_labels)

        loss = nll(predictions, output_labels.view(svhn.N_LABELS * batch_size))
        loss.backward()
        sgd.step()

        losses.append(loss.data[0])
        running_loss = float(sum(losses)) / float(len(losses))
        print('Loss = {:.3f} at {}/{}.'.format(running_loss, i, n_iters),
              end='\r')
