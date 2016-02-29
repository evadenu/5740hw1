package nlp.assignments;

import java.util.List;
import nlp.assignments.MaximumEntropyClassifier.EncodedDatum;
import nlp.assignments.MaximumEntropyClassifier.Encoding;
import nlp.assignments.MaximumEntropyClassifier.IndexLinearizer;
import nlp.classify.BasicFeatureVector;
import nlp.classify.BasicLabeledFeatureVector;
import nlp.classify.FeatureExtractor;
import nlp.classify.FeatureVector;
import nlp.classify.LabeledFeatureVector;
import nlp.classify.LabeledInstance;
import nlp.math.DoubleArrays;
import nlp.util.Counter;
import nlp.util.Indexer;

public class PerceptronClassifier<I, F, L> {

	double[] w;
	int iterations;
	FeatureExtractor<I, F> featureExtractor;
	Encoding<F, L> encoding;
	IndexLinearizer indexLinearizer;
	
	public PerceptronClassifier(FeatureExtractor<I, F> fe) {
		featureExtractor = fe;
	}
	
	public void trainPerceptron(List<LabeledInstance<I, L>> trainingData, boolean average){
		System.out.println("Encoding data...");
		// build classifier
		// build data encodings so the inner loops can be efficient
		encoding = buildEncoding(trainingData);
		indexLinearizer = buildIndexLinearizer(encoding);
		double[] w = buildInitialWeights(indexLinearizer);
		double[] w_avg = buildInitialWeights(indexLinearizer);
		EncodedDatum[] data = encodeData(trainingData, encoding);
		
		System.out.print("Training perceptron...");

		//Loop through w and adjust if incorrect
		//y*=argmax_y_wy * O(x)
		
//		Counter<Integer> labels = new Counter<Integer>();
//		Counter<Integer> plabels = new Counter<Integer>();
		
		for(int i = 0; i < data.length; i++){
			int pred = predict(data[i], indexLinearizer, w);
			int trueLabel = data[i].getLabelIndex();
			
//			labels.incrementCount(trueLabel, 1.0);
//			plabels.incrementCount(pred, 1.0);
			
			if(pred != trueLabel) {
				w = updateWeights(i, data[i], w, pred, trueLabel, indexLinearizer, 1);
			}
			if(average) {
				w_avg = vec_add(w_avg, w);
			}
		}
		
//		for(int l: labels.keySet()) {
//			System.out.println(l + " " + labels.getCount(l) + " " + plabels.getCount(l));
//		}
		
		if(average) {
			this.w = vec_div(w_avg, data.length);
		} else {
			this.w = w;
		}
		System.out.println("Done.");
	}
	
	/*
	 * Add vectors element-wise
	 */
	private double[] vec_add(double[] v1, double[] v2) {
		assert v1.length == v2.length;
		for(int i = 0; i < v1.length; i++) {
			v1[i] += v2[i];
		}
		return v1;
	}
	
	/*
	 * Divide vector by value element-wise
	 */
	private double[] vec_div(double[] v, double val) {
		for(int i = 0; i < v.length; i++) {
			v[i] /= val;
		}
		return v;
	}
	
	/*
	 * Update weight vector by subtracting predicted features and adding true label features
	 */
	private double[] updateWeights(int i, EncodedDatum datum, double[] w, int predLabel, int trueLabel, IndexLinearizer indexLinearizer, int sign) {
		for(int n = 0; n < datum.getNumActiveFeatures(); n++) {
			// Subtract predicted features
			if(predLabel != -1) {
				int pLinIndex = indexLinearizer.getLinearIndex(datum.getFeatureIndex(n), predLabel);
				w[pLinIndex] -= datum.getFeatureCount(n);
			}

			// Add true features
			int tLinIndex = indexLinearizer.getLinearIndex(datum.getFeatureIndex(n), trueLabel);
			w[tLinIndex] += datum.getFeatureCount(n);
		}
		return w;
	}
	
	/*
	 * Compute argmax y, given a new sample and weight vector
	 */
	private int predict(EncodedDatum datum, IndexLinearizer indexLinearizer, double[] w) {
		int max_label = -1;
		double max_score = 0;
		
		for(int i = 0; i < indexLinearizer.numLabels; i++) {
			double score = 0;
			
			for(int n = 0; n < datum.getNumActiveFeatures(); n++) {
				int linIndex = indexLinearizer.getLinearIndex(datum.getFeatureIndex(n), i);
				score += w[linIndex] * datum.getFeatureCount(n);
			}
			
			if(score > max_score) {
				max_score = score;
				max_label = i;
			}
		}
		return max_label;
	}
	
	/*
	 * Predict a label for the given name
	 */
	public L getLabel(I name) {
		FeatureVector<F> featureVector = new BasicFeatureVector<F>(
				featureExtractor.extractFeatures(name));
		EncodedDatum encodedDatum = EncodedDatum.encodeDatum(featureVector,
				encoding);
		
		int pred = predict(encodedDatum, indexLinearizer, w);
		return encoding.getLabel(pred);
	}
	
	/*
	 * Copied from MaximumEntropyClassifier
	 */
	private double[] buildInitialWeights(IndexLinearizer indexLinearizer) {
		return DoubleArrays.constantArray(0.0,
				indexLinearizer.getNumLinearIndexes());
	}

	/*
	 * Copied from MaximumEntropyClassifier
	 */
	private IndexLinearizer buildIndexLinearizer(Encoding<F, L> encoding) {
		return new IndexLinearizer(encoding.getNumFeatures(),
				encoding.getNumLabels());
	}

	/*
	 * Copied from MaximumEntropyClassifier
	 */
	public Encoding<F, L> buildEncoding(List<LabeledInstance<I, L>> data) {
		Indexer<F> featureIndexer = new Indexer<F>();
		Indexer<L> labelIndexer = new Indexer<L>();
		for (LabeledInstance<I, L> labeledInstance : data) {
			L label = labeledInstance.getLabel();
			Counter<F> features = featureExtractor
					.extractFeatures(labeledInstance.getInput());
			LabeledFeatureVector<F, L> labeledDatum = new BasicLabeledFeatureVector<F, L>(
					label, features);
			labelIndexer.add(labeledDatum.getLabel());
			for (F feature : labeledDatum.getFeatures().keySet()) {
				featureIndexer.add(feature);
			}
		}
		return new Encoding<F, L>(featureIndexer, labelIndexer);
	}

	/*
	 * Copied from MaximumEntropyClassifier
	 */
	private EncodedDatum[] encodeData(List<LabeledInstance<I, L>> data,
			Encoding<F, L> encoding) {
		EncodedDatum[] encodedData = new EncodedDatum[data.size()];
		for (int i = 0; i < data.size(); i++) {
			LabeledInstance<I, L> labeledInstance = data.get(i);
			L label = labeledInstance.getLabel();
			Counter<F> features = featureExtractor
					.extractFeatures(labeledInstance.getInput());
			LabeledFeatureVector<F, L> labeledFeatureVector = new BasicLabeledFeatureVector<F, L>(
					label, features);
			encodedData[i] = EncodedDatum.encodeLabeledDatum(
					labeledFeatureVector, encoding);
		}
		return encodedData;
	}

}
