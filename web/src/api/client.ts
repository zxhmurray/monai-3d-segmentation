/**
 * API 客户端
 */

import axios from 'axios';
import type {
  HealthResponse,
  ModelInfo,
  InferenceResult,
  BatchInferenceResult,
  VolumeReport,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 分钟超时（推理可能需要较长时间）
});

// 健康检查
export const healthCheck = async (): Promise<HealthResponse> => {
  const response = await apiClient.get<HealthResponse>('/health');
  return response.data;
};

// 模型列表
export const listModels = async (): Promise<string[]> => {
  const response = await apiClient.get<string[]>('/models');
  return response.data;
};

// 模型信息
export const getModelInfo = async (modelName: string): Promise<ModelInfo> => {
  const response = await apiClient.get<ModelInfo>(`/models/${modelName}`);
  return response.data;
};

// 加载模型
export const loadModel = async (modelName: string): Promise<void> => {
  await apiClient.post(`/models/${modelName}/load`);
};

// 卸载模型
export const unloadModel = async (modelName: string): Promise<void> => {
  await apiClient.post(`/models/${modelName}/unload`);
};

// 已加载模型列表
export const getLoadedModels = async (): Promise<string[]> => {
  const response = await apiClient.get<string[]>('/loaded_models');
  return response.data;
};

// 单图像推理
export const predict = async (
  file: File,
  config: {
    modelName?: string;
    threshold?: number;
    overlap?: number;
    returnProb?: boolean;
  }
): Promise<InferenceResult> => {
  const formData = new FormData();
  formData.append('file', file);

  if (config.modelName) formData.append('model_name', config.modelName);
  if (config.threshold !== undefined) formData.append('threshold', config.threshold.toString());
  if (config.overlap !== undefined) formData.append('overlap', config.overlap.toString());
  if (config.returnProb !== undefined) formData.append('return_prob', config.returnProb.toString());

  const response = await apiClient.post<InferenceResult>('/inference/predict', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

// 批量推理
export const predictBatch = async (
  filePaths: string[],
  config: {
    modelName?: string;
    threshold?: number;
    overlap?: number;
  }
): Promise<BatchInferenceResult> => {
  const response = await apiClient.post<BatchInferenceResult>('/inference/predict_batch', {
    file_paths: filePaths,
    model_name: config.modelName,
    threshold: config.threshold,
    overlap: config.overlap,
  });

  return response.data;
};

// 获取体积报告
export const getVolumeReport = async (caseId: string): Promise<VolumeReport> => {
  const response = await apiClient.get<VolumeReport>(`/inference/volume/${caseId}`);
  return response.data;
};

export default apiClient;
