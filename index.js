import dotenv from "dotenv";
dotenv.config();

import { StateGraph, START, END, MemorySaver } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import * as z from "zod";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

const CONFIG = {
  maxRetries: 3,
  baseURL: "https://api.longcat.chat/openai",
  timeout: 30000,
};

const outputSchema = z.object({
  continuation: z.string().describe("The newly written continuation of the script"),
  continuation_summary: z.string().describe("Brief summary of the new continuation"),
  completed: z.boolean().describe("Whether this is the final part of the script"),
  script_type: z.enum(["intro", "body", "conclusion", "hook"]).optional().describe("Type of content generated"),
});

const State = z.object({
  topic: z.string(),
  max_words: z.number().min(10).max(40000),
  word_count: z.number().min(0),
  word_remaining: z.number(),
  script: z.string(),
  summary: z.string(),
  completed: z.boolean(),
  script_type: z.string().optional(),
  iterations: z.number().min(0).default(0),
  error_count: z.number().min(0).default(0),
});

class ScriptWriterAgent {
  constructor() {
    this.model = new ChatOpenAI({
      apiKey: process.env.LLM_API_KEY,
      model: process.env.LLM_MODEL_NAME,
      configuration: {
        baseURL: CONFIG.baseURL,
      },
      timeout: CONFIG.timeout,
    });

    this.modelWithStructuredOutput = this.model.withStructuredOutput(outputSchema);
    this.agent = this.buildAgent();
  }

  getSystemPrompt() {
    return new SystemMessage(`You are a professional YouTube script writer.

CRITICAL RULES:
- STRICTLY adhere to word limits. NEVER exceed Words Remaining.
- Write engaging, conversational content suitable for YouTube.
- Maintain consistent tone and style throughout.
- When approaching word limit, wrap up naturally.
- Mark as completed when:
  * Word count is very close to maximum
  * Content feels complete and satisfying
  * Natural conclusion point is reached

CONTENT TYPES:
- Hook: Captivating opening (first segment only)
- Intro: Introduce topic and what viewers will learn
- Body: Main content with key points
- Conclusion: Summary and call-to-action

Response format:
{
  "continuation": "The script continuation",
  "continuation_summary": "Brief summary",
  "completed": true/false,
  "script_type": "hook|intro|body|conclusion"
}`);
  }

  async llmCall(state) {
    try {
      const response = await this.modelWithStructuredOutput.invoke([
        this.getSystemPrompt(),
        new HumanMessage(this.buildHumanMessage(state))
      ]);

      return this.processResponse(state, response);
    } catch (error) {
      console.error("LLM call failed:", error);
      return this.handleError(state, error);
    }
  }

  buildHumanMessage(state) {
    const isFirstIteration = state.word_count === 0;
    
    return `Please write the ${isFirstIteration ? 'first part' : 'next part'} of a YouTube script:

Topic: ${state.topic}
Current Word Count: ${state.word_count}
Maximum Word Limit: ${state.max_words}
*** WORDS REMAINING: ${state.word_remaining} *** (ABSOLUTE LIMIT!)
${!isFirstIteration ? `Summary So Far: ${state.summary}` : 'This is the beginning of the script.'}

${isFirstIteration ? 'Start with an engaging hook to capture attention immediately.' : 'Continue naturally from the existing content.'}`;
  }

  processResponse(state, response) {
    const newWords = response.continuation.split(/\s+/).length;
    const newWordCount = state.word_count + newWords;
    const wordsRemaining = state.max_words - newWordCount;

    const actualContinuation = wordsRemaining < 0 
      ? response.continuation.split(/\s+/).slice(0, state.word_remaining).join(' ')
      : response.continuation;

    const actualWordCount = wordsRemaining < 0 ? state.max_words : newWordCount;

    const newState = {
      ...state,
      word_count: actualWordCount,
      word_remaining: Math.max(0, wordsRemaining),
      script: state.script + actualContinuation + " ",
      summary: state.summary + response.continuation_summary + " ",
      completed: response.completed || wordsRemaining <= 50,
      script_type: response.script_type,
      iterations: state.iterations + 1,
    };

    this.logProgress(state, newState, newWords);
    return newState;
  }

  handleError(state, error) {
    const newState = {
      ...state,
      error_count: state.error_count + 1,
    };

    if (newState.error_count >= CONFIG.maxRetries) {
      newState.completed = true;
      console.error("Max retries exceeded, forcing completion");
    }

    return newState;
  }

  logProgress(oldState, newState, newWords) {
    console.log(`   Iteration ${newState.iterations}:`);
    console.log(`   Added ${newWords} words | Total: ${newState.word_count}/${newState.max_words}`);
    console.log(`   Remaining: ${newState.word_remaining} words | Completed: ${newState.completed}`);
    console.log(`   Type: ${newState.script_type || 'N/A'}`);
    console.log('---');
  }

  shouldContinue(state) {
    if (state.completed || state.word_remaining <= 0) {
      return END;
    }
    if (state.iterations > 20) { 
      console.warn("Max iterations reached, forcing completion");
      return END;
    }
    return "llmCall";
  }

  buildAgent() {
    return new StateGraph(State)
      .addNode("llmCall", (state) => this.llmCall(state))
      .addEdge(START, "llmCall")
      .addConditionalEdges("llmCall", (state) => this.shouldContinue(state), {
        "llmCall": "llmCall",
        [END]: END
      })
      .compile();
  }

  async generateScript(topic, maxWords = 1000) {
    console.log(`Starting script generation: "${topic}" (${maxWords} words max)`);
    
    const initialState = {
      topic,
      max_words: maxWords,
      word_count: 0,
      word_remaining: maxWords,
      script: "",
      summary: "",
      completed: false,
      iterations: 0,
      error_count: 0,
    };

    const result = await this.agent.invoke(initialState);
    
    console.log(`   Script completed!`);
    console.log(`   Final: ${result.word_count}/${maxWords} words`);
    console.log(`   Iterations: ${result.iterations}`);
    
    return {
      script: result.script.trim(),
      metadata: {
        wordCount: result.word_count,
        iterations: result.iterations,
        finalType: result.script_type,
        efficiency: Math.round((result.word_count / maxWords) * 100)
      }
    };
  }
}

export async function generateYouTubeScript(topic, maxWords = 1000) {
  try {
    const agent = new ScriptWriterAgent();
    const result = await agent.generateScript(topic, maxWords);
    
    return {
      success: true,
      ...result
    };
  } catch (error) {
    console.error("Script generation failed:", error);
    return {
      success: false,
      error: error.message,
      script: "",
      metadata: {}
    };
  }
}

export async function generateScriptWithOptions(options) {
  const { topic, maxWords = 1000, style = "conversational" } = options;
  
  const styledTopic = style === "professional" 
    ? `${topic} - Present in a professional, educational style`
    : style === "entertaining" 
    ? `${topic} - Make it entertaining and humorous`
    : topic;

  return await generateYouTubeScript(styledTopic, maxWords);
}