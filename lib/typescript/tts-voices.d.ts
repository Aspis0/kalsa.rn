export interface OuteTTSWord {
    word: string;
    duration: number;
    codes: number[];
}
export interface OuteTTSSpeaker {
    words: OuteTTSWord[];
}
export interface NeuTTSSpeaker {
    ref_phones: string;
    ref_codes: number[];
}
export type SpeakerPayload = OuteTTSSpeaker | NeuTTSSpeaker | {
    text?: string;
    [k: string]: any;
};
export declare function lookupVoice(family: string, name: string, language?: string): SpeakerPayload | null;
export declare function listVoices(family: string, language?: string): string[];
export declare function listLanguages(family: string): string[];
//# sourceMappingURL=tts-voices.d.ts.map