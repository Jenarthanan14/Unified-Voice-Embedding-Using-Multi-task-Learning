
def get_embeddings(self, cluster_count=None, stored=False, total_embeddings=40):
    if stored:
        #print("Running in store mode : ")
        return self.get_stored_embeddings(cluster_count, total_embeddings=total_embeddings)
    else:
        return self.create_and_get_embeddings(cluster_count)

def get_stored_embeddings(self, cluster_count=None, total_embeddings=40):
    set_of_embeddings = []
    set_of_speakers = []
    speaker_numbers = []

    set_of_embeddings.append(self.embeddings_dict['set_of_embeddings'][0][0:cluster_count] +
                             self.embeddings_dict['set_of_embeddings'][0][total_embeddings:cluster_count + total_embeddings])
    set_of_speakers.append(self.embeddings_dict['set_of_speakers'][0][0:cluster_count] +
                           self.embeddings_dict['set_of_speakers'][0][total_embeddings:cluster_count + total_embeddings])
    speaker_numbers.append(self.embeddings_dict['speaker_numbers'])

    return None, set_of_embeddings, set_of_speakers, speaker_numbers

def create_and_get_embeddings(self, cluster_count=None, stored=False, total_embeddings=40):
	logger = get_logger('lstm', logging.INFO)
	logger.info('Run pairwise_lstm test')

	x_test, speakers_test = load_and_prepare_data(self.get_validation_test_data(), cluster_count=cluster_count * 2)
	x_train, speakers_train = load_and_prepare_data(self.get_validation_train_data(), cluster_count=cluster_count * 8)

	set_of_embeddings = []
	set_of_speakers = []
	speaker_numbers = []
	#checkpoints = list_all_files(get_experiment_nets(), "*pairwise_lstm*.h5")
	#checkpoints = ["pairwise_lstm_100_00999.h5"]

	# Values out of the loop
	#metrics = ['accuracy', 'categorical_accuracy', ]less_speaker_test_1_2
	#loss = pairwise_kl_divergence
	#custom_objects = {'pairwise_kl_divergence': pairwise_kl_divergence}
	#optimizer = 'rmsprop'
	vector_size = 256 * 2

	# Fill return values
	for checkpoint in self.checkpoints:
	    logger.info('Running checkpoint: ' + checkpoint)
	    # Load and compile the trained network
	    #network_file = get_experiment_nets(checkpoint)
	    #model_full = load_model(network_file, custom_objects=custom_objects)
	    #model_full.compile(loss=loss, optimizer=optimizer, metrics=metrics)

	    # Get a Model with the embedding layer as output and predict
	    #model_partial = Model(inputs=model_full.input, outputs=model_full.layers[2].output)
	    model_partial = self.model_dict[checkpoint]
	    test_output = np.asarray(model_partial.predict(x_test))
	    # print("------------------>>>>>>>>>>> test data size\n")
	    # print(x_test.shape)
	    #
	    # print("------------------>>>>>>>>>>> prediction out\n")
	    # print(test_output.shape)
	    train_output = np.asarray(model_partial.predict(x_train))
	    #
	    # print("------------------>>>>>>>>>>> train data size\n")
	    # print(x_train.shape)
	    #
	    # print("------------------>>>>>>>>>>> prediction out\n")
	    # print(train_output.shape)

	    embeddings, speakers, num_embeddings = generate_embeddings(train_output, test_output, speakers_train,
	                                                               speakers_test, vector_size)
	    set_of_embeddings.append(embeddings)
	    set_of_speakers.append(speakers)
	    speaker_numbers.append(num_embeddings)

	    #return 1

	logger.info('Pairwise_lstm test done.')
	return self.checkpoints, set_of_embeddings, set_of_speakers, speaker_numbers

def generate_embeddings(self, train_output, test_output, train_speakers, test_speakers, vector_size, sr):
    """
    Combines the utterances of the speakers in the train- and testing-set and combines them into embeddings.
    :param train_output: The training output (8 sentences)
    :param test_output:  The testing output (2 sentences)
    :param train_speakers: The speakers used in training
    :param test_speakers: The speakers used in testing
    :param vector_size: The size which the output will have
    :return: embeddings, the speakers and the number of embeddings
    """
    logger = get_logger('clustering', logging.INFO)
    logger.info('Generate embeddings')
    num_speakers = len(set(test_speakers))

    # Prepare return variable
    number_embeddings = 2 * num_speakers
    embeddings = []
    speakers = []

    # Create utterances
    embeddings_train, speakers_train = self.create_utterances(num_speakers, vector_size, train_output,
                                                              train_speakers, sr)
    embeddings_test, speakers_test = self.create_utterances(num_speakers, vector_size, test_output, test_speakers,
                                                            sr)

    # Merge utterances
    embeddings.extend(embeddings_train)
    embeddings.extend(embeddings_test)
    speakers.extend(speakers_train)
    speakers.extend(speakers_test)

    # print(print("------------------>>>>>>>>>>> embeddings and speakers\n"))
    # print(embeddings)
    # print(speakers)

    return embeddings, speakers, number_embeddings

def create_utterances(self, num_speakers, vector_size, vectors, speaker_list, sr):
    """
    Creates one utterance for each speaker in the vectors.
    :param num_speakers: Number of distinct speakers in this vector
    :param vector_size: Number of data in utterance
    :param vectors: The unordered speaker data
    :param y: An array that tells which speaker (number) is in which place of the vectors array
    :return: the embeddings per speaker and the speakers (numbers)
    """

    # Prepare return variables
    dimension_reduction = DIMENSION_REDUCTION

    embeddings = np.zeros((num_speakers, vector_size))
    buckets = build_buckets(MAX_SEC, BUCKET_STEP, FRAME_STEP)
    speakers = set(speaker_list)

    # Fill embeddings with utterances
    for i in range(num_speakers):
        print("Generating embeddings : ", i + 1, "/", num_speakers)
        # Fetch correct utterance
        flatten_embeddings = embeddings[i]

        # Fetch values where same speaker and add to utterance
        indices = np.where(speaker_list == i)[0]
        # print("-------------->>>>>>>>>>>>>> indices ", indices)
        outputs = np.take(vectors, indices, axis=0)
        time_series = None
        for j in range(len(outputs)):
            # print("------------------------>>>>>>>>>>>>>>>> debug timeseries extraction")
            # print(j)
            if j == 0:
                time_series = np.trim_zeros(outputs[j, 0, 0], 'b')
                # print(len(time_series))
            else:
                time_series = np.concatenate((time_series, np.trim_zeros(outputs[j, 0, 0], 'b')))

        # print("------------------------>>>>>>>>>>>>>>>> time_series data embeddings")
        # print(time_series)
        # print(len(time_series))

        mfcc = get_fft_spectrum_from_data(time_series, buckets)

        #print(mfcc.shape)
        concept_embeddings = np.squeeze(self.model_partial.predict(mfcc.reshape(1, *mfcc.shape, 1)))
        #print("concept embeddings : ",  concept_embeddings)
        # if dimension_reduction:
        #     for k in range(len(concept_embeddings)):
        #         for l in range(concept_embeddings.shape[1]):
        #             for m in range(concept_embeddings.shape[2]):
        #                 flatten_embeddings = np.add(flatten_embeddings, concept_embeddings[k][l][m])
        #
        #     divider_length = concept_embeddings.shape[0] * concept_embeddings.shape[1] * concept_embeddings.shape[2]
        #     # Add filled utterance to embeddings
        #     embeddings[i] = np.divide(flatten_embeddings, divider_length)
        # else:
        #     for k in range(len(concept_embeddings)):
        #         flatten_embeddings = np.add(flatten_embeddings, concept_embeddings[k])

        #embeddings[i] = np.divide(flatten_embeddings, len(concept_embeddings))
        embeddings[i] = concept_embeddings
        #print(embeddings[i])

    return embeddings, speakers


def load_and_prepare_data(data_path, segment_size=50, cluster_count=None):
    # Load and generate test data
    x, y, s_list,sr = load(data_path)

    if cluster_count is not None:
        x = x[0:cluster_count]
        y = y[0:cluster_count]
        s_list = s_list[0:cluster_count]


    # x, speakers = generate_test_data(def storeDict(filename, dict_to_store):
    with open(get_experiment_results(filename), 'wb') as f:
        pickle.dump(dict_to_store, f, -1)x, y, segment_size)

    # Reshape test data because it is an lstm
    return x, y,sr

def storeDict(filename, dict_to_store):
    with open(get_experiment_results(filename), 'wb') as f:
        pickle.dump(dict_to_store, f, -1)





def test_network(self, cluster_count=None, mr_list=None, algorithm='AHC', stored=False, total_embeddings=40, epsilon=1e-6, cutoff=-0.1):
    """
    Tests the network implementation with the validation data set and saves the result sets
    of the different metrics in analysis.
    """

    checkpoint_names, set_of_predicted_clusters, set_of_true_clusters, embeddings_numbers = self\
        .get_clusters(cluster_count, algorithm, stored, total_embeddings=total_embeddings, epsilon=epsilon,
                      cutoff=cutoff)
    network_name = self.name + '_' + self.val_data
    checkpoint_names = [algorithm]
    analyse_results(network_name, checkpoint_names, set_of_predicted_clusters, set_of_true_clusters,
                    embeddings_numbers, cluster_count, mr_list)



def get_clusters(self, cluster_count=None, algorithm='AHC', stored=False, total_embeddings=40,
	             epsilon=1e-6, cutoff=-0.1):
	"""
	Generates the predicted_clusters with the results of get_embeddings.
	All return values are sets of possible multiples.
	:return:
	checkpoint_names: A list of names from the checkpoints. Later used as curvenames,
	set_of_predicted_clusters: A 2D array of the predicted Clusters from the Network. [checkpoint, clusters]
	set_of_true_clusters: A 2d array of the validation clusters. [checkpoint, validation-clusters]
	embeddings_numbers: A list which represent the number of embeddings in each checkpoint.
	"""

	print("Incoming couster count inside get clusters ===========> ", cluster_count)
	checkpoint_names, set_of_embeddings, set_of_true_clusters, embeddings_numbers = self.get_embeddings(cluster_count, stored,
	                                                                                                    total_embeddings=total_embeddings)

	print("Before clustering ===========> ", cluster_count)
	set_of_predicted_clusters = cluster_embeddings(set_of_embeddings, cluster_count=cluster_count,
	                                               algorithm=algorithm, set_of_speakers=set_of_true_clusters,
	                                               epsilon=epsilon, cutoff=cutoff)

	#print("------------------>>>>>>>>>>> predicted results\n")
	#print(set_of_predicted_clusters)

	#print("------------------>>>>>>>>>>> true clusters\n")
	#print(set_of_true_clusters)

	return checkpoint_names, set_of_predicted_clusters, set_of_true_clusters, embeddings_numbers

