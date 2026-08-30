package dev.logickoder.keyguarde.app.service

import android.content.Context
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.text.textembedder.TextEmbedder
import kotlin.collections.forEach
import kotlin.math.sqrt


class AISemanticMatcher(
    private val context: Context
) {
    private var textEmbedder: TextEmbedder? = null
    private val keywordEmbeddingCache = mutableMapOf<String, FloatArray>()

    fun setupTextEmbedder() {
        val baseOptionsBuilder = BaseOptions.builder()
        val baseOption =
            baseOptionsBuilder.setModelAssetPath("universal_sentence_encoder.tflite").build()
        val options = TextEmbedder.TextEmbedderOptions.builder().setBaseOptions(baseOption).build()

        textEmbedder = TextEmbedder.createFromOptions(context, options)
    }

   private fun embedText(word: String): FloatArray? {
        return textEmbedder?.let { textEmbedder ->
            textEmbedder.embed(word)
                .embeddingResult()
                .embeddings()[0]
                .floatEmbedding()
        }
    }

    /**
     * Calc the nearness of 2 vectors:
     * similarity = 1 -> vectors point same direction; same
     * similarity = 0 -> vectors perpendicular; unrelated
     * similarity = -1 -> vectors point opposite directions; opposite meaning
     * Cosine similarity: DotProduct of the vectors / Cross Product of Magnitude of vectors
     */
    private fun cosineSimilarity(vectorX: FloatArray, vectorY: FloatArray): Float {
        var dotProduct = 0.0
        var normX = 0.0
        var normY = 0.0

        vectorX.forEachIndexed { index, _ ->
            dotProduct += vectorX[index] * vectorY[index]

            normX += vectorX[index] * vectorX[index]
            normY += vectorY[index] * vectorY[index] //squares
        }

        val magX: Double = sqrt(normX)
        val magY: Double = sqrt(normY)

        val similarity = dotProduct / (magX * magY)
        return similarity.toFloat()
    }

   fun findSimilarWords(userKeywords: List<String>, incomingText: String): List<String> {
        val incomingVector = embedText(incomingText) //Embed the notification text once for all keywords;
        val similarWords = mutableListOf<String>()

        userKeywords.forEach { keyWord ->
            var vectorSimilarity = 0.0f

            if (keywordEmbeddingCache.keys.contains(keyWord)) {
                if (incomingVector != null) {
                    keywordEmbeddingCache[keyWord]?.let {
                        vectorSimilarity = cosineSimilarity(
                            vectorX = it,
                            vectorY = incomingVector
                        )
                    }
                }
                if (vectorSimilarity > SIMILARITY_THRESHOLD) {
                    similarWords.add(keyWord)
                }
            } else {
                val keywordVector = embedText(keyWord) //would run for thr first time when cache is empty

                if (keywordVector != null && incomingVector != null) {
                    keywordEmbeddingCache[keyWord] = keywordVector
                    vectorSimilarity = cosineSimilarity(
                        vectorX = keywordVector,
                        vectorY = incomingVector
                    )
                }

                if (vectorSimilarity > SIMILARITY_THRESHOLD) {
                    similarWords.add(keyWord)
                }
            }
        }

        return similarWords
    }


    fun clearTextEmbedder() {
        textEmbedder?.close()
        textEmbedder = null
    }

    /**
     * Ensure the embedded keywords in cache
     * are for the updated keywords put by user by:-
     * removing Deleted Words From Vector Map Cache [keywordEmbeddingCache]
     */
    fun syncCache(currentKeywords: List<String>) {
        val currentKeywordsSet = currentKeywords.toSet()
        keywordEmbeddingCache.keys.retainAll(currentKeywordsSet)
    }

    companion object {
        const val SIMILARITY_THRESHOLD = 0.65f
    }

}