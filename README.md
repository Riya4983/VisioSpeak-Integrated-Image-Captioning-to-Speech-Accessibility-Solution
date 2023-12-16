VisioSpeak – Integrated Image Captioning – to – Speech Accessibility Solution


Problem Statement

This project aims to develop a computer vision system that can automatically generate captions for images, enabling visually impaired individuals to better understand their surroundings. Visually impaired individuals face inherent challenges in comprehending and interacting with visual information, especially in a world that heavily relies on visual content. The primary question this project seeks to answer is: How can we empower visually impaired individuals by creating a system that interprets and communicates the content of images in a clear and actionable manner?

One of the fundamental issues is the limited accessibility of visual content, as it often relies on graphical information that is not easily translatable into non-visual forms. This poses a significant barrier for individuals with visual impairments in understanding the environment around them, hindering their independence and overall quality of life. The project aims to bridge this gap by developing an intelligent system that can automatically generate descriptive captions for images, providing a more comprehensive understanding of visual scenes.

To address this problem, the project employs a combination of computer vision and natural language processing techniques. Image features are extracted using the VGG16 model, and a deep learning architecture is employed for caption generation. The integration with OpenAI's GPT-3.5 API further enhances the system's ability to provide coherent and context-aware responses to user interactions.

The significance of this problem statement lies in its potential to significantly improve the daily lives of visually impaired individuals by granting them access to visual information in a manner that is both informative and actionable. The project aims to contribute to the broader field of assistive technology, fostering inclusivity and independence for individuals with visual impairments in an increasingly visual world.

Data Sources

The project utilizes two primary data sources:

Images: The dataset consists of images downloaded from Flickr, focusing on diverse scenes such as landscapes, objects, and people. The specific details of the dataset, including its size and distribution, are not provided in the code.

Captions: Each image is associated with corresponding captions manually annotated by humans. These captions provide textual descriptions of the image content and serve as the ground truth for training the caption generation model.
Link:  Flickr dataset
Data Preparation

The data preparation process involves the following steps:
Cleaning: Textual captions are cleaned by removing noise and irrelevant characters, such as punctuation and special symbols.

Preprocessing: Captions are further preprocessed by applying techniques like lowercase conversion and tokenization. Tokens are then padded to ensure consistent input sequences for the model.
Feature Extraction: Features are extracted from each image using the pre-trained VGG16 model. These features capture the visual characteristics of the image and are used as input to the caption generation model.
Partitioning: The data is partitioned into training and test sets, ensuring a 90:10 split, and a generator function is employed to handle data in batches during model training. The training set is used to train the model and tune the hyperparameters and the test set is used to evaluate the final model performance.

Data Modelling

A) Image Captioning: Our project transcends traditional image captioning, empowering visually impaired individuals through advanced technology. Seamless integration of cutting-edge image captioning models enhances comprehension of visual content, leveraging descriptive AI for accurate and contextually rich captions. This innovation serves as a crucial bridge between visual and textual realms, meeting the unique needs of visually impaired individuals. A noteworthy advancement, our project signifies a substantial leap in leveraging AI to amplify accessibility, enriching experiences for the visually impaired community.
 
Inputs: 
•	Images: Loaded and preprocessed using VGG16.
•	Captions: Extracted from a text file, preprocessed for tokenization.
Outputs:
•	Image Features: VGG16-extracted features stored in a dictionary.
•	Tokenized Captions: Preprocessed captions ready for training.
•	Captioning Model: Trained model capable of generating captions for input images.
Tools used:
•	VGG16 Model: TensorFlow Keras pre-trained model for image feature extraction.
•	Tokenizer: Keras Tokenizer for text preprocessing.
•	Embedding Layer: Part of the captioning model for textual representation.
•	LSTM Layer: Sequential processing in the decoder of the captioning model.
•	Categorical Cross entropy Loss: Used during model compilation.
•	Adam Optimizer: Optimizer for weight updates during model training.

 
 B) GPT 3.5 Enabled Actionable Directives: Utilizing GPT-3.5, actionable directives are seamlessly implemented for image classification. GPT-3.5's advanced capabilities facilitate the generation of precise directives with contextual relevance, enhancing user guidance through intricate visual scenarios. The utilization of GPT-3.5 underscores the commitment to deploying cutting-edge solutions in accessibility and user experience.
Inputs: 
•	System Message: Guides GPT-3.5 on the reasoning process.
•	User Messages: Include image captions for contextual understanding.
Outputs:
GPT-3.5 Responses: 
Responses are generated providing actionable directives.
The algorithm used for achieving the mentioned objective is as follows:
•	Receive Input:
Accept input describing a blind person's perception.
•	Check for Actionability:
Determine if the input contains actionable information.
•	If non-actionable:
Generate concise output for non-actionable input.
Output format: "Two kids are playing." or "People are having food."
•	If Actionable:
If the input requires action (e.g., "a car is coming"), proceed to the next step.
•	Generate Actionable Output:
Provide a suggestion for the probable action the blind person should take based on the input.
Output format: "Stop for a while until the car passes." or "Listen carefully for approaching traffic."
•	Include Action and Description:
Ensure the output clearly mentions the action to be taken and provides a brief description.
Output format: "Action: Stop. Description: A car is approaching."
•	Final Output:
Present the final output as a clear and concise directive, emphasizing the recommended action for the blind person.

Tools used:
•	OpenAI GPT-3.5 API: Utilized for natural language understanding and response generation.
•	Prompt Engineering: System and user messages structured to guide GPT-3.5's responses.

 

C) Empowering visually impaired with text-to-speech guidance: Utilizing sophisticated algorithms, this system converts written content into auditory experiences, delivering precise guidance tailored for visually impaired users. By translating text into clear and concise verbal directives, our system enables a more intuitive and enriching user experience, bridging the gap between the visual and auditory domains. This innovative integration marks a significant stride towards empowering visually impaired individuals with enhanced tools for information comprehension and interaction.
Input:
•	The text content to be converted to speech.
•	The language in which the text should be converted (e.g., 'en' for English).
Output:
•	Audio File: The converted audio is saved as an MP3 file in the specified working directory.
•	Playing the converted audio file.
Tools Used:
•	gTTS: This is used for text-to-speech conversion.
•	Os: This module is used for interacting with the operating system, in this case, for saving the converted audio file.
•	Playsound: Used to play the converted audio file.

Model Evaluation
 
We critically evaluated output performance, analyzing key metrics for each solution.
1.	Image captioning:
1.1	Model training:
Inputs:
Image features extracted using VGG16.
Tokenized captions preprocessed for training.
Outputs:
Trained model with optimized parameters.
Metrics: Categorical cross entropy Loss, Adam Optimizer.
Results: Successful convergence of the model with minimized loss.
1.2	Model prediction:
Inputs: Actual captions from the test set.
Outputs: Predicted captions generated by the model.
Metrics:
•	BLEU-1 score: This suggests that, on average, over half of the unigrams in the generated text match with those in the reference text.
•	BLEU-2 score: This indicates that about 29% of bigrams in the generated text align with those in the reference text.
Results:  BLEU-1 score: 0.537292,  BLEU-2 score: 0.288973
2.	GPT 3.5 enabled actionable directives: 

2.1	Prompt Effectiveness:

Inputs:
•	System message instructing GPT-3.5.
•	User messages including image captions.

Outputs: GPT-3.5 responses providing actionable directives.
Metrics: Qualitative assessment of directive clarity.
Results: Responses demonstrate effective guidance for users.

2.2	User Interaction Quality:

Inputs: Integration of GPT-3.5 responses into user interactions.
Outputs: Coherent and context-aware user interactions.
Metrics: Subjective assessment of user interaction quality.
Results: Natural and meaningful interactions enhancing user experience.
3.	Empowering visually impaired with text-to-speech guidance: For evaluating the text-to-speech output the following metric can be further implemented:
Mean Opinion Score (MOS): MOS is a subjective measure where human listeners rate the naturalness of synthesized speech on a scale. A group of listeners evaluate the quality of the speech, and their scores are averaged.

Recommendations
 
To optimize the intelligent image captioning system for the empowerment of visually impaired individuals, it is crucial to implement a multi-faceted approach. First and foremost, fine-tuning the model by diversifying training datasets and exploring alternative pre-trained models will enhance its adaptability to diverse real-world scenarios. Simultaneously, refining the integration of GPT-3.5 through precise instructions and varied prompts will optimize natural language interactions, providing clearer and more context-aware responses. Augmenting user experience through additional accessibility features, such as voice commands and real-time image recognition, will ensure a seamless and inclusive interaction for visually impaired users. Establishing a continuous feedback loop for iterative system improvement, along with collaboration with assistive technology developers, will contribute to a more comprehensive and integrated solution within the broader accessibility ecosystem. User education and outreach efforts are essential to increase awareness and adoption, fostering valuable feedback for ongoing refinement. Commitment to these recommendations will undoubtedly propel the intelligent image captioning system towards its full potential as a transformative assistive technology, positively impacting the lives of visually impaired individuals.

