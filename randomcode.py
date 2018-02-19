



reshapedimage = img
reshapedimage = cv2.resize(img,(299, 299), interpolation = cv2.INTER_CUBIC)
transform = transforms.ToTensor()
# transformedimage = transform(reshapedimage.astype(float))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
# preprocess = transforms.Compose([
#     transforms.Scale(299),
#     transforms.ToTensor(),
#     normalize
#     ])
preprocess = transforms.Compose([
    # transforms.Scale(299,299),
    transforms.ToTensor(),
    normalize
    ])

transformedimage = preprocess(reshapedimage)
image_variable = Variable(transformedimage, requires_grad = True)
image_variable = image_variable.float()

image_variable = image_variable.unsqueeze(0)

img_input = torch.autograd.Variable(image_variable.data.cpu() , requires_grad = True)


img_output = a.inceptionfeaturesmodel(img_input)


img_output.backward(gradient=torch.ones(img_input.size(), retain_variables=True)

grad_img_input = img_input.grad



# fig = plt.figure()
# """
# ax1 = fig.add_subplot(121)
# # Bilinear interpolation - this will look blurry
# ax1.imshow(maxfilt, cmap=cm.Greys_r)
# """
#
# """
# ax2 = fig.add_subplot(122)
# # 'nearest' interpolation - faithful but blocky
# ax2.imshow(maxfilt, interpolation='nearest', cmap=cm.Greys_r)
# """
#
#
# ax1 = fig.add_subplot(121)
# ax1.imshow(maxfilt1, interpolation='nearest', cmap=cm.Greys_r)
#
# ax2 = fig.add_subplot(122)
# ax2.imshow(maxfilt2, interpolation='nearest', cmap=cm.Greys_r)
#
# plt.show()
