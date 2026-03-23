/**
 * API 类型定义
 */

export interface HealthResponse {
  status: string;
  version: string;
  timestamp: string;
}

export interface ModelInfo {
  name: string;
  version: string;
  device: string;
  loaded: boolean;
  trained_epochs?: number;
  best_dice?: number;
}

export interface InferenceResult {
  case_id: string;
  status: 'success' | 'failed';
  message: string;
  prediction_path?: string;
  dice_score?: number;
  volume_cm3?: number;
  voxel_count?: number;
  processing_time: number;
}

export interface BatchInferenceResult {
  total: number;
  success: number;
  failed: number;
  results: InferenceResult[];
}

export interface VolumeReport {
  case_id: string;
  voxel_count: number;
  volume_mm3: number;
  volume_cm3: number;
  spacing: [number, number, number];
}

export interface InferenceConfig {
  modelName?: string;
  threshold: number;
  overlap: number;
  returnProb: boolean;
}

export interface UploadedFile {
  id: string;
  name: string;
  size: number;
  status: 'pending' | 'uploading' | 'processing' | 'done' | 'error';
  progress?: number;
  result?: InferenceResult;
  error?: string;
}

export interface HistoryItem {
  id: string;
  fileName: string;
  uploadTime: string;
  status: 'success' | 'failed';
  result?: InferenceResult;
}
