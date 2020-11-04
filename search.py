from usrlib.document import Document, read_corpus, calc_collection_frequency
from usrlib.invertedindex import InvertedIndex
import usrlib.boolean_retrieval as boolean_retrieval
import usrlib.vector_space as vector_space
import os, time, pickle
from pandas import read_pickle

#user query function
def user_query(vsmodel, corpus, df, index):
    use_boolean = True
    while True:
        query = input("Enter query: ")
        if query == "EXIT":
            break
        else:
            if use_boolean:
                print("\nBoolean Retrieval results: ")
                start = time.time()
                try:
                    output = boolean_retrieval.parse_query(query, corpus, index)
                
                    for fileid in output:
                        print(corpus[fileid].filepath)
                    end = time.time()
                    print(len(output),"files returned in", end-start, 's')
                
                except:
                    print("Query doesn't match any documents for boolean retrival")	
                
            
            else:
                output = [i.doc_id for i in corpus]

            print("\nTf-Idf results: ")
            start = time.time()
            
            try:
                output = vector_space.parse_query(query, corpus, vsmodel, df, output)
            
                for file, prob in output[:10]:
                    print(file, "\t", prob)
                end = time.time()
                print("returned in ", end-start, 's')
                
            except:
                print("Query doesn't match any documents for vector space model")

if __name__=='__main__':

    print("\n***Program started***\n")

    if("df.pickle" in os.listdir("./saved_files")) and ("corpus.pickle" in os.listdir("./saved_files")) and ("inv_index.pickle" in os.listdir("./saved_files")):
        # folder name is corpus in this case

        # loading corpus
        start = time.time()
        with open("./saved_files/corpus.pickle", "rb") as corpus_file:
            corpus = pickle.load(corpus_file)
        end = time.time()
        print("corpus loaded in: "+str(end - start))        
        
        # loading inverted index object
        start = time.time()
        with open("./saved_files/inv_index.pickle", "rb") as inv_index_file:
            index = pickle.load(inv_index_file)
        end = time.time()
        print("inverted index loaded in: "+str(end - start))        

        # loading vectorspace object
        vsmodel = vector_space.Tf_Idf(inv_index=index)
        start = time.time()
        df = read_pickle('./saved_files/df.pickle')
        end = time.time()
        print("dataframe pickle loaded in: "+str(end - start))        

    
    else:
        print("reading files")
        start = time.time()
        corpus = read_corpus('corpus')
        end = time.time()
        print("corpus construction time: "+str(end - start)+"s")

        # saving corpus object
        with  open("./saved_files/corpus.pickle", "wb") as corpus_file:
            pickle.dump(corpus, corpus_file)

        collection_freq = calc_collection_frequency(corpus)

        print("Constructing inverted index.....")
        start = time.time()
        index = InvertedIndex(corpus)
        end = time.time()
        print("Total inverted index construction time: "+str(end - start))
        # saving inverted index object
        with open("./saved_files/inv_index.pickle", "wb") as inv_index_file:
            pickle.dump(index, inv_index_file)

        print("Constructing vector space model........")
        start = time.time()
        vsmodel = vector_space.Tf_Idf(inv_index=index)
        # saving tf idf dataframe object
        df = vsmodel.get_dataframe(corpus, collection_freq)
        df.to_pickle('./saved_files/df.pickle')
        end = time.time()


        print("Total vector space model construction time: "+str(end - start)+"s")

    print('Dataframe size: ', df.shape[0], 'tokens X', df.shape[1], 'docs')
    user_query(vsmodel, corpus, df, index)
    
    #Program end
