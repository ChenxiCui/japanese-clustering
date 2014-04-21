package japanseseclusterer;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import org.apache.mahout.common.Pair;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.utils.clustering.ClusterDumper;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.ja.JapaneseAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.util.Version;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.clustering.classify.WeightedVectorWritable;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.vectorizer.DictionaryVectorizer;
import org.apache.mahout.vectorizer.DocumentProcessor;
import org.apache.mahout.vectorizer.SparseVectorsFromSequenceFiles;
import org.apache.mahout.vectorizer.tfidf.TFIDFConverter;
import org.apache.mahout.math.NamedVector;
import org.atilika.kuromoji.Token;
import org.atilika.kuromoji.Tokenizer;

/**
 * Clusters Japanese sentences according to topics 
 */
public class JapaneseSentenceClusterer {
		
	private Configuration conf = new Configuration();
	
	private FileSystem fs;
	
	private Path inputFile;
	
	private Path outputDir;
	
	private Path seqFilesDir;
	
	private Path clustersDir;
	
	private Path vectorsDir;
	
	private Path tfidfVectorsDir;
	
	private Path resultsDir;
	
	/** 
	 * Constructor - initializes FileSystem and clears the output folder
	 * 
	 * @param inputFile Path to file with one sentence per line
	 * @param outputDir Path to output directory
	 */
	public JapaneseSentenceClusterer(Path inputFile, Path outputDir) throws IOException {
		
		this.inputFile = inputFile;
		this.outputDir = outputDir;
		
		this.seqFilesDir = new Path(outputDir, "seqfiles");
		this.clustersDir = new Path(outputDir, "clusters"); 
		this.vectorsDir = new Path(outputDir, "vectors"); 
		this.tfidfVectorsDir = new Path(outputDir, "tfidf-vectors");
		this.resultsDir = new Path(outputDir, "results");
		
		try {
			this.fs = FileSystem.get(conf);
			// delete old files
			fs.delete(outputDir, true);
		} catch (IOException e) {
			throw new IOException("File error while creating new Clusterer", e);
		}
		
	}
	
	/**
	 * Starts all steps from input files to clustering
	 * 
	 * @return true on successfull run
	 * @throws Exception 
	 */
	public void run() throws Exception {
		List<String> sentencesText;
		
		sentencesText = readFile();
		processText(sentencesText);   // Tokenize, filter nouns, sparse vectors (tf-id ...)
		cluster();
		
	}
	
	/**
	 * Reads the input file and stores the lines in an array
	 */
	private List<String> readFile() throws IOException {
				
		String line;
				
		List<String> sentences = new ArrayList<String>();
		
		try (BufferedReader br = Files.newBufferedReader(Paths.get(inputFile.toString()), Charset.forName("UTF-8"))) {
			int lineNumber = 1;
			while ((line = br.readLine()) != null) {
				
				sentences.add(line);
				lineNumber++;
				if(lineNumber > 100000) {	// TODO: remove after testing
					break;
				}
			}

			br.close();
		} catch (IOException e) {
			throw new IOException("Error while reading input file "+e.getMessage(), e);
		}

		return sentences;		
		
	}
	
	/**
	 * Tokenizes the sentences with Kuromoji and saves them as sequence files.
	 * Then generates vectors (tf, tfidf), dictionary etc.
	 * @throws IOException, Exception 
	 */
	private void processText(List<String> sentences) throws IOException, Exception {
		
		try (SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, seqFilesDir, Text.class, Text.class)) {
					
			int sentenceNum = 0;
			
			// write raw text to sequencefiles
			for (String rawSentence : sentences) {
				writer.append(new Text("sentence"+sentenceNum++), new Text(rawSentence));
			}
			writer.close();	
			
			JapaneseAnalyzer analyzer = new JapaneseAnalyzer(Version.LUCENE_47);
			
			Path tokenizedDocsPath = new Path(outputDir, DocumentProcessor.TOKENIZED_DOCUMENT_OUTPUT_FOLDER);
			
			DocumentProcessor.tokenizeDocuments(seqFilesDir, analyzer.getClass().asSubclass(Analyzer.class), tokenizedDocsPath, conf);
			
			int minSupport = 2;
			int minDf = 5;
			int maxDFPercent = 95;
			int maxNGramSize = 2;
			int minLLRValue = 50;
			int reduceTasks = 1;
			int chunkSize = 200;
			int norm = 2;
			boolean sequentialAccessOutput = true;
			boolean logNormalize = true;
			
			DictionaryVectorizer.createTermFrequencyVectors(tokenizedDocsPath, outputDir, DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER, conf, minSupport, maxNGramSize, minLLRValue, 
					2, true, reduceTasks, chunkSize, sequentialAccessOutput, true);
						
			Pair<Long[], List<Path>> features = TFIDFConverter.calculateDF(new Path(outputDir.toString(), DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER), outputDir, conf, chunkSize);
			
			TFIDFConverter.processTfIdf(new Path(outputDir.toString(), DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER), outputDir, conf, features, minDf, maxDFPercent, norm, logNormalize, sequentialAccessOutput, true, reduceTasks);
			
			// TODO: add normalization ?		
			
		} catch(IOException e) {
			throw new IOException("File error while processing text. File: "+e.getMessage(), e);
		} 
		
	}
	
	/**
	 * Does the actual clustering using k-means with an initial random seed
	 * @throws IOException 
	 * @throws InterruptedException 
	 * @throws ClassNotFoundException 
	 */
	private void cluster() throws IOException, ClassNotFoundException, InterruptedException {	
		
			RandomSeedGenerator.buildRandom(conf, tfidfVectorsDir, clustersDir, 10, new EuclideanDistanceMeasure());
			
			/* better use canopy clustering for cluster initialization?
 			Path canopyCentroids = new Path(outputDir, "canopy-centroids");
			CanopyDriver.run(conf, tfidfVectorsDir, canopyCentroids, new EuclideanDistanceMeasure(), 250, 120, true, 0.5, false);
			 */
			
			// params: conf, input path, cluster path, output path, convergenceDelta, maxIterations, run clustering, classification threshold, runsequential
			KMeansDriver.run(conf, tfidfVectorsDir, clustersDir, resultsDir, 0.001, 10, true, 0.5, true);  // change last true to false for hadoop
	
	}
	
	/**
	 * Prints the assignment of sentences to clusters (work in progress)
	 * (Alternatively use Mahout's clusterdumper tool, which shows the top terms per cluster etc.)
	 * 
	 */
	private void printResults() throws IOException {
		
		ClusterDumper dumper = new ClusterDumper(seqFilesDir, new Path(resultsDir + "/" + Cluster.CLUSTERED_POINTS_DIR + "/part-m-0"));
		
		// TODO: set the clusters-*-final path automatically. Right now the clusterdumper does not output anything if the no of iterations does not match the value here
		String[] params = {"-dt", "sequencefile", "-d", "data/output/dictionary.file-*", "-i", "data/output/results/clusters-2-final/", "-p", "data/output/results/clusteredPoints"};
		
		try {
			System.out.println("Running clusterdumper: ");
			dumper.run(params);
		} catch (Exception e1) {
			System.err.println("Error while running clusterdumper");
			e1.printStackTrace();
		}
		
		/* TODO: fix Reader
		try {
			SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path(resultsDir + "/" + Cluster.CLUSTERED_POINTS_DIR + "/part-m-0"), conf);
			IntWritable key = new IntWritable();
			WeightedPropertyVectorWritable value = new WeightedPropertyVectorWritable();
			
			// cluster number, sentences id
			HashMap<Integer, List<String>> cluster = new HashMap<Integer, List<String>>();
			
			while (reader.next(key, value)) {

				NamedVector vec = (NamedVector) value.getVector();
				
				// create entry for cluster, if it does not exist yet
				if(!cluster.containsKey(key.get())) { 
					cluster.put(key.get(), new ArrayList<String>());
				}
				
				cluster.get(key.get()).add(vec.getName());

			}
			reader.close();
			
			Iterator iterator = cluster.keySet().iterator();
			while(iterator.hasNext()) {
				Integer clusterId = (Integer)iterator.next();
				System.out.println("Cluster "+clusterId);
				List<String> sentences = (List<String>) cluster.get(clusterId);
				
				for(String sentence : sentences) {
					System.out.print(sentence+" ");  // TODO: print the sentence itself, instead of just the id
				}
				System.out.println("\n");
			}
			
		} catch (IOException e) {
			throw new IOException("File error while reading result. File: " + e.getMessage(), e);
		}
		*/
	}
	
	public static void main(String[] args) {
				
		try {
			JapaneseSentenceClusterer clusterer = new JapaneseSentenceClusterer(new Path("data/input/examples.utf"), new Path("data/output"));
					
			clusterer.run();
	
			// check result
			clusterer.printResults();
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}

}