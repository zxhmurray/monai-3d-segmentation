/**
 * 全局状态管理 (Zustand)
 */

import { create } from 'zustand';
import type { UploadedFile, HistoryItem, InferenceConfig, ModelInfo } from '../types';

interface AppState {
  // 上传文件
  uploadedFiles: UploadedFile[];
  addFile: (file: UploadedFile) => void;
  updateFile: (id: string, updates: Partial<UploadedFile>) => void;
  removeFile: (id: string) => void;
  clearFiles: () => void;

  // 推理配置
  inferenceConfig: InferenceConfig;
  setInferenceConfig: (config: Partial<InferenceConfig>) => void;

  // 历史记录
  history: HistoryItem[];
  addHistory: (item: HistoryItem) => void;
  clearHistory: () => void;

  // 模型
  availableModels: string[];
  loadedModels: string[];
  selectedModel: string | null;
  setAvailableModels: (models: string[]) => void;
  setLoadedModels: (models: string[]) => void;
  setSelectedModel: (model: string | null) => void;

  // UI 状态
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
}

export const useStore = create<AppState>((set) => ({
  // 上传文件
  uploadedFiles: [],
  addFile: (file) => set((state) => ({
    uploadedFiles: [...state.uploadedFiles, file],
  })),
  updateFile: (id, updates) => set((state) => ({
    uploadedFiles: state.uploadedFiles.map((f) =>
      f.id === id ? { ...f, ...updates } : f
    ),
  })),
  removeFile: (id) => set((state) => ({
    uploadedFiles: state.uploadedFiles.filter((f) => f.id !== id),
  })),
  clearFiles: () => set({ uploadedFiles: [] }),

  // 推理配置
  inferenceConfig: {
    threshold: 0.5,
    overlap: 0.5,
    returnProb: false,
  },
  setInferenceConfig: (config) => set((state) => ({
    inferenceConfig: { ...state.inferenceConfig, ...config },
  })),

  // 历史记录
  history: [],
  addHistory: (item) => set((state) => ({
    history: [item, ...state.history].slice(0, 100), // 最多保留 100 条
  })),
  clearHistory: () => set({ history: [] }),

  // 模型
  availableModels: [],
  loadedModels: [],
  selectedModel: null,
  setAvailableModels: (models) => set({ availableModels: models }),
  setLoadedModels: (models) => set({ loadedModels: models }),
  setSelectedModel: (model) => set({ selectedModel: model }),

  // UI 状态
  isLoading: false,
  setIsLoading: (loading) => set({ isLoading: loading }),
}));
