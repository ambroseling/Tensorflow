# Tensorflow Developer Certificate Repository
This is a repository for all the code I wrote to prepare for the Tensorflow developer certificate exam

## What is the TensorFlow Developer Certificate?
The TensorFlow Developer Certificate is a certification program offered by Google that tests your understanding on how to build and train deep learning models in Tensorflow. You can check out their website for more details about the certificate itself, in short it is a test of your foundational knowledge on ML in Tensorflow

## Why take the Tensorflow Developer Certificate Exam?
I personally took the TensorFlow Developer Certificate because its a good way to practice your skills in building deep learning models ranging in areas from regression, computer vision, image classification, object detection, sentiment analysis, text classification and time-series analysis. Another thing is that it is a demonstration of competency and it could be great for when you're taking on a ML role or you're starting out in this field and you want to learn more about the most popular ML framework,this is something that can definitely serve you well and be useful.

## Exam format?
The exam is 100USD, allowed time is 5 hours with 5 questions. Each question increases in difficulty and requires you to train, save and submit a model to be marked out of 5. The exam also happens in PyCharm through a Tensorflow Developer Certificate Plugin. Areas of the question range from regression, computer vision, image classification, object detection, sentiment analysis, text classification, text generation and time-series analysis. So being prepared on all fronts is very imporant. If you make sure to do the preparation, you should have plenty of time to finish the 5 questions. You are allowed to any resource at your disposal (Google, Colab, Books, Your Notes). 

## How to prepare for the exam?
First step is definitely reading over the candidate handbook, making sure that you are comfortable with all the topics covered. It also covers some more technical details about the exam as well as how to set up your environment ready for the exam and making sure you can install the proper dependencies in your environment.

For the exam, I would suggest doing the exam on a computer that has access to a GPU. For most of the models a GPU isn' required or very beneficial but it definitely would speed up the training process by a significant portion. I personally used a windows laptop that had a GPU to take the exam but I also have a Mac and I experimented and I did try looking into taking the exam on my Mac but Tensorflow has mentioned that M1 support has not beeen officially provided so I do not recomment taking it on Mac with Apple Silicon M1 or M2 chips. I also have not been able to configure tensorflow-metal for my Mac that is just for my personal tensorflow environment, so I recommend to just take the exam on a non-Mac computer. You should definitely take advantage of Colab as you have limited access to GPU usage but you at least can train multiple models at once to speed up your exam.

## My experience taking the exam?
I was first exposed to the exam in first year of university when I came across an online course on Udemy by Daniel Bourke called the TensorFlow Developer Certificate: Zero to Mastery. I got really interested in Tensorflow and found it incredibly simple to build models for a variety of tasks. I started learning from the course around February of 2022 until April. However at the time it was during school and I had other courses to focus on so as soon as I was done the course, I took a break and didn't think about the exam. Summer of 2023 hits and I recall taking the course and decided to set myself a goal to take the exam in July. My revision started in end of May 2023 by reviewing the course material, doing through all the practice and coding problems of each section in the course. In parallel I got myself 2 books [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python) and the [Hands On Machine Learning with Scikit-Learn, Keras and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/). Both these book are amazing in covering the fundamentals and the detailed architectures commonly used in Computer Vision, NLP and Time Series.I personally made notes, tried out the examples in the book, tried doing the exercises in the books as well. Then in other times I prepared by finding random datasets online to practice from Kaggle. This is a good way of being comfortable dealing with different data sources, whether it be csv, json, txt files. 

Closer to the exam date that I decided to practice by finding a dataset and timing yourself to simulate the exam environment. I think this might not be necessary if you personally are very comfortable with building the models quickly. I also just spent a lot of time reviewing key components that are useful when training models. Things like using callbacks (helps with converging to a solution with higher accuracy, could be through ReduceLRonPlateau, ModelCheckPoints, EarlyStopping, LearningRateScheduler), data augmentation (for image classification tasks that could help reduce overfitting, making your model generalize better when there's more variability), embeddings and tokenizers (knowing how to quickly turn words or characters into tokens and using tokenized forms of data for training)

I think the most challenging part is deciding to do the exam. Since this is all self-motivated it is very easy to procrastinate or push the exam to another day. But I think it is important to be able to pick a day and sit down and actually do the exam.

Keep in mind I had a lot of other commitments during my time of revising for the exam so it took me a lot longer to be fully ready for the exam. But I guess nobody is really fully ready for something until you do it, so if you put in the time and the effort, you could be well-prepared in a shorter amount of time.

## Volume 1: Fundamentals (Simple Classification & Regression)


## Volume 1: Fundamentals (Simple Classification & Regression)

## Volume 2: Computer Vision 

## Volume 3: Natural Language Processing

## Volume 4: Time Series & Sequential Analysis
