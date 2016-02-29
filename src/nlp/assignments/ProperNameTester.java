package nlp.assignments;

import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import nlp.classify.*;
import nlp.util.CommandLineUtils;
import nlp.util.Counter;

/**
 * This is the main harness for assignment 2. To run this harness, use
 * <p/>
 * java nlp.assignments.ProperNameTester -path ASSIGNMENT_DATA_PATH -model
 * MODEL_DESCRIPTOR_STRING
 * <p/>
 * First verify that the data can be read on your system using the baseline
 * model. Second, find the point in the main method (near the bottom) where a
 * MostFrequentLabelClassifier is constructed. You will be writing new
 * implementations of the ProbabilisticClassifer interface and constructing them
 * there.
 */
public class ProperNameTester {

	public static class ProperNameFeatureExtractor implements
			FeatureExtractor<String, String> {

		/*
		 * Extract n-gram features for a name
		 */
		public void extractNGrams(String name, int i, Counter<String> features, int n) {
			for(int ng = 0; ng < n; ng++) {
				if(i + ng + 1 > name.length()) break;
				features.incrementCount(ng + "-" + name.substring(i, i + ng + 1), 1.0);
			}
		}
		
		/*
		 * Extract non-ngram features for a name
		 */
		public void extractOtherFeatures(char[] characters, char ch, int i, Counter<String> features) {
			if(i < characters.length-1) {
				char ch1 = characters[i+1];
				if(ch == '\'' && ch1 == 's') {
					features.incrementCount("CONTAINS-POSS", 10.0);
				}
			}
			
			if(i < characters.length-2) {
				char ch1 = characters[i+1];
				char ch2 = characters[i+2];

				if(ch == 'I' && ch1 == 'n' && ch2 == 'c') {
					features.incrementCount("INCORPORATED", 5.0);
				}
				if(ch == 'T' && characters[i+1] == 'h' && characters[i+2] == 'e') {
					features.incrementCount("CONTAINS-THE", 10.0);
				}
			}
			
			if(ch == '/') {
				features.incrementCount("CONTAINS-SLASH", 50.0);
			}
			
			if(ch == ':') {
				features.incrementCount("CONTAINS-COLON", 50.0);
			}
			
			if(ch == ' ') {
				features.incrementCount("NUM-WORDS", 1.0);
			}
			
			if("1234567890".contains(ch + "")) {
				features.incrementCount("NUMBERS", 1.0);
			}
		}
		
		/**
		 * This method takes the list of characters representing the proper name
		 * description, and produces a list of features which represent that
		 * description. The basic implementation is that only character-unigram
		 * features are extracted. An easy extension would be to also throw
		 * character bigrams into the feature list, but better features are also
		 * possible.
		 */
		public Counter<String> extractFeatures(String name) {
			// append start and end indicators
			name = "<" + name + ">";
			
			char[] characters = name.toCharArray();
			Counter<String> features = new Counter<String>();
			
			// word bigram features
			String[] tokens = name.split(" ");
			for(int i = 0; i < tokens.length; i++) {
				features.incrementCount("WORD-" + tokens[i], 1.0);
				if(i < tokens.length - 1) {
					features.incrementCount("WORD-" + tokens[i] + tokens[i+1], 1.0);
				}
			}
			
			int nGrams = 12;
			
			for (int i = 0; i < characters.length; i++) {
				char ch = characters[i];
				
				// add n-gram features
				extractNGrams(name, i, features, nGrams);
				
				// Extract all other features
				extractOtherFeatures(characters, ch, i, features);
			}
			
			return features;
		}
	}

	private static List<LabeledInstance<String, String>> loadData(
			String fileName) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(fileName));
		List<LabeledInstance<String, String>> labeledInstances = new ArrayList<LabeledInstance<String, String>>();
		while (reader.ready()) {
			String line = reader.readLine();
			String[] parts = line.split("\t");
			String label = parts[0];
			String name = parts[1];
			LabeledInstance<String, String> labeledInstance = new LabeledInstance<String, String>(
					label, name);
			labeledInstances.add(labeledInstance);
		}
		reader.close();
		return labeledInstances;
	}
	
	private static double max(ArrayList<Double> arr) {
		double max = 0;
		for(double d: arr) {
			if(d > max) {
				max = d;
			}
		}
		return max;
	}

	private static double mean(ArrayList<Double> arr) {
		double total = 0;
		for(double d: arr) {
			total += d;
		}
		return total/arr.size();
	}
	
	private static void testClassifier(
			ProbabilisticClassifier<String, String> classifier, PerceptronClassifier<String, String, String> pClassifier,
			List<LabeledInstance<String, String>> testData, boolean verbose, boolean usePerceptron) {
		double numCorrect = 0.0;
		double numTotal = 0.0;
		
		// Confidence scores and confusion matrix
		ArrayList<Double> confidences = new ArrayList<Double>();	
		HashMap<Double, Integer[]> confidence_scores = new HashMap<Double, Integer[]>();
		HashMap<String, Counter<String>> confusion = new HashMap<String, Counter<String>>();
		
		for (LabeledInstance<String, String> testDatum : testData) {
			String name = testDatum.getInput();
			String predLabel = null;
			double confidence = 0;
			
			if(usePerceptron) {
				predLabel = pClassifier.getLabel(name);
			} else {
				predLabel = classifier.getLabel(name);
				confidence = classifier.getProbabilities(name).getCount(
						predLabel);
			}
			
			String truthLabel = testDatum.getLabel();
			
			// Add to confidence scores
			confidences.add(confidence);
			
			double rounded_conf = Math.round(confidence * 10)/10.0;
			if(!confidence_scores.containsKey(rounded_conf)) {
				Integer[] arr = {0, 0};
				confidence_scores.put(rounded_conf, arr);
			}
			Integer[] counts = confidence_scores.get(rounded_conf);
			counts[1]++;	// update total count for this score
			
			// Update confusion matrix
			if(!confusion.containsKey(truthLabel)) {
				confusion.put(truthLabel, new Counter<String>());
			}
			confusion.get(truthLabel).incrementCount(predLabel, 1.0);

			if (predLabel.equals(truthLabel)) {
				numCorrect += 1.0;
				counts[0]++;	// update total correct for this conf score
			} else {
				if(confidence >= 0.8) {
					System.out.println("Misclassified: " + name + " as " + predLabel + " ; " + truthLabel);
				}
				if (verbose) {
					// display an error
					System.err.println("Error: " + name + " guess=" + predLabel
							+ " gold=" + testDatum.getLabel() + " confidence="
							+ confidence);
				}
			}
			numTotal += 1.0;
		}
		double accuracy = numCorrect / numTotal;
		
		// Analyze confusion matrix
		for(String tLabel: confusion.keySet()) {
			System.out.print(tLabel);
			double total = 0;
			double correct = 0;
			Counter<String> predCounter = confusion.get(tLabel);
			for(String pLabel: confusion.keySet()) {
				double count = predCounter.getCount(pLabel);
				
				if(pLabel.equals(tLabel)) {
					correct += count;
				}
				total += count;
				System.out.print(" " + (int) count);
			}
			System.out.println(" Acc: " + (correct/total));
		}
		
		// Analyze confidence score
		for(double score: confidence_scores.keySet()) {
			Integer[] counts = confidence_scores.get(score);
			System.out.println(score + " " + (double) counts[0]/counts[1]);
		}
		
		System.out.println("Accuracy: " + accuracy);
		System.out.println("max conf: " + max(confidences) + " mean: " + mean(confidences));
	}

	public static void main(String[] args) throws IOException {
		// Parse command line flags and arguments
		Map<String, String> argMap = CommandLineUtils
				.simpleCommandLineParser(args);

		// Set up default parameters and settings
		String basePath = ".";
		String model = "baseline";
		boolean verbose = false;
		boolean useValidation = true;
		boolean shuffle = false;
		boolean averagePerceptron = false;
		
		// Update defaults using command line specifications

		// The path to the assignment data
		if (argMap.containsKey("-path")) {
			basePath = argMap.get("-path");
		}
		System.out.println("Using base path: " + basePath);

		// A string descriptor of the model to use
		if (argMap.containsKey("-model")) {
			model = argMap.get("-model");
		}
		System.out.println("Using model: " + model);

		// A string descriptor of the model to use
		if (argMap.containsKey("-test")) {
			String testString = argMap.get("-test");
			if (testString.equalsIgnoreCase("test"))
				useValidation = false;
		}
		System.out.println("Testing on: "
				+ (useValidation ? "validation" : "test"));

		// Whether or not to print the individual speech errors.
		if (argMap.containsKey("-verbose")) {
			verbose = true;
		}
		
		// Whether or not to shuffle the training data lines
		if (argMap.containsKey("-shuffle")) {
			shuffle = true;
		}
		if (argMap.containsKey("-average")) {
			averagePerceptron = true;
		}

		// Load training, validation, and test data
		List<LabeledInstance<String, String>> trainingData = loadData(basePath
				+ "/pnp-train" + (shuffle? "-shuffle" : "") + ".txt");
		List<LabeledInstance<String, String>> validationData = loadData(basePath
				+ "/pnp-validate.txt");
		List<LabeledInstance<String, String>> testData = loadData(basePath
				+ "/pnp-test.txt");

		// Learn a classifier
		ProbabilisticClassifier<String, String> classifier = null;
		PerceptronClassifier<String, String, String> perceptronClassifier = null;
		boolean usePerceptron = false;
		
		if (model.equalsIgnoreCase("baseline")) {
			classifier = new MostFrequentLabelClassifier.Factory<String, String>()
					.trainClassifier(trainingData);
		} else if (model.equalsIgnoreCase("n-gram")) {
			// TODO: construct your n-gram model here
			
			
			
			
		} else if (model.equalsIgnoreCase("maxent")) {
			// Train max entropy classifier
			ProbabilisticClassifierFactory<String, String> factory = new MaximumEntropyClassifier.Factory<String, String, String>(
					1.0, 40, new ProperNameFeatureExtractor());
			classifier = factory.trainClassifier(trainingData);
		} else if (model.equalsIgnoreCase("percep")) {
			usePerceptron = true;
			
			// Train perceptron classifier
			perceptronClassifier = new PerceptronClassifier<String, String, String>(new ProperNameFeatureExtractor());
			perceptronClassifier.trainPerceptron(trainingData, averagePerceptron);
			
		} else {
			throw new RuntimeException("Unknown model descriptor: " + model);
		}

		// Test classifier
		testClassifier(classifier, perceptronClassifier, (useValidation ? validationData : testData),
				verbose, usePerceptron);
	}
}
