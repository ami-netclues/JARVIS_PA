// Minimal types for Web Speech API (not included in TS DOM lib by default)

export {};

declare global {
type SpeechRecognitionEvent = Event & {
  results: SpeechRecognitionResultList;
};

interface SpeechRecognition extends EventTarget {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  start(): void;
  stop(): void;
  onresult: ((this: SpeechRecognition, ev: SpeechRecognitionEvent) => any) | null;
  onerror: ((this: SpeechRecognition, ev: Event) => any) | null;
}
}

