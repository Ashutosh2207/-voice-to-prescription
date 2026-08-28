import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';

export async function transcribeAudio(audioPath) {
  const pythonScript = `
import whisper
import sys

model = whisper.load_model("base")
result = model.transcribe("${audioPath.replace(/\\/g, '\\\\')}")
print(result["text"])
`;

  const tempScript = path.join(process.cwd(), 'temp_transcribe.py');
  fs.writeFileSync(tempScript, pythonScript);

  try {
    const output = execSync(`python ${tempScript}`, { encoding: 'utf-8', timeout: 120000 });
    fs.unlinkSync(tempScript);
    return output.trim();
  } catch (error) {
    fs.unlinkSync(tempScript);
    throw new Error(`Transcription failed: ${error.message}`);
  }
}

export async function transcribeWithOpenAI(audioPath, apiKey) {
  const { default: OpenAI } = await import('openai');
  const fs = await import('fs');
  
  const openai = new OpenAI({ apiKey });
  const transcription = await openai.audio.transcriptions.create({
    file: fs.createReadStream(audioPath),
    model: 'whisper-1',
    language: 'en',
  });
  
  return transcription.text;
}