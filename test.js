import { startAgent } from "./index.js";

const TOPIC = "The fox and the cunning crow";
const MAX_WORDS = 5000;

const response = await startAgent(TOPIC, MAX_WORDS);

console.log("SCRIPT:\n", response.script);