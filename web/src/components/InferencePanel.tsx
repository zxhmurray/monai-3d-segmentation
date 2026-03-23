/**
 * 推理配置与执行面板
 */

import React from 'react';
import { Play, Settings, CheckCircle, XCircle, Loader2 } from 'lucide-react';
import { useStore } from '../hooks/useStore';
import { predict } from '../api/client';
import type { UploadedFile, HistoryItem } from '../types';

export const InferencePanel: React.FC = () => {
  const {
    uploadedFiles,
    updateFile,
    clearFiles,
    inferenceConfig,
    setInferenceConfig,
    selectedModel,
    addHistory,
  } = useStore();

  const pendingFiles = uploadedFiles.filter((f) => f.status === 'pending');
  const isProcessing = uploadedFiles.some((f) => f.status === 'uploading' || f.status === 'processing');

  const handleInference = async () => {
    if (pendingFiles.length === 0) return;

    for (const file of pendingFiles) {
      // 更新状态为上传中
      updateFile(file.id, { status: 'uploading', progress: 0 });

      try {
        updateFile(file.id, { status: 'processing' });

        // 调用推理 API
        const result = await predict(
          new File([], file.name), // 实际使用时应传递真实文件
          {
            modelName: selectedModel || undefined,
            threshold: inferenceConfig.threshold,
            overlap: inferenceConfig.overlap,
            returnProb: inferenceConfig.returnProb,
          }
        );

        // 更新文件状态
        updateFile(file.id, { status: 'done', result });

        // 添加到历史记录
        const historyItem: HistoryItem = {
          id: file.id,
          fileName: file.name,
          uploadTime: new Date().toISOString(),
          status: result.status as 'success' | 'failed',
          result,
        };
        addHistory(historyItem);

      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : '推理失败';
        updateFile(file.id, {
          status: 'error',
          error: errorMessage,
        });

        // 添加到历史记录（失败）
        const historyItem: HistoryItem = {
          id: file.id,
          fileName: file.name,
          uploadTime: new Date().toISOString(),
          status: 'failed',
        };
        addHistory(historyItem);
      }
    }
  };

  const getStatusIcon = (status: UploadedFile['status']) => {
    switch (status) {
      case 'done':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'error':
        return <XCircle className="h-5 w-5 text-red-500" />;
      case 'uploading':
      case 'processing':
        return <Loader2 className="h-5 w-5 text-blue-500 animate-spin" />;
      default:
        return null;
    }
  };

  return (
    <div className="space-y-6">
      {/* 配置区域 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
        <div className="flex items-center space-x-2 mb-4">
          <Settings className="h-5 w-5 text-gray-500" />
          <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100">
            推理配置
          </h3>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-gray-500 mb-1">
              阈值 (Threshold)
            </label>
            <input
              type="number"
              min="0"
              max="1"
              step="0.1"
              value={inferenceConfig.threshold}
              onChange={(e) => setInferenceConfig({ threshold: parseFloat(e.target.value) })}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md
                         bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100
                         focus:ring-2 focus:ring-blue-500 focus:outline-none"
            />
          </div>

          <div>
            <label className="block text-xs text-gray-500 mb-1">
              重叠率 (Overlap)
            </label>
            <input
              type="number"
              min="0"
              max="1"
              step="0.1"
              value={inferenceConfig.overlap}
              onChange={(e) => setInferenceConfig({ overlap: parseFloat(e.target.value) })}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md
                         bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100
                         focus:ring-2 focus:ring-blue-500 focus:outline-none"
            />
          </div>
        </div>
      </div>

      {/* 文件状态列表 */}
      {uploadedFiles.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
          <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-3">
            文件状态
          </h3>
          <ul className="space-y-2">
            {uploadedFiles.map((file) => (
              <li
                key={file.id}
                className="flex items-center justify-between py-2 border-b border-gray-100 dark:border-gray-700 last:border-0"
              >
                <div className="flex items-center space-x-3">
                  {getStatusIcon(file.status)}
                  <span className="text-sm text-gray-700 dark:text-gray-300">
                    {file.name}
                  </span>
                </div>
                {file.result && (
                  <span className="text-xs text-gray-500">
                    Dice: {file.result.dice_score?.toFixed(4) || 'N/A'}
                  </span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* 执行按钮 */}
      <div className="flex space-x-3">
        <button
          onClick={handleInference}
          disabled={pendingFiles.length === 0 || isProcessing}
          className={`
            flex-1 flex items-center justify-center space-x-2 px-4 py-3 rounded-lg
            font-medium text-white transition-colors
            ${pendingFiles.length === 0 || isProcessing
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-700'
            }
          `}
        >
          {isProcessing ? (
            <>
              <Loader2 className="h-5 w-5 animate-spin" />
              <span>处理中...</span>
            </>
          ) : (
            <>
              <Play className="h-5 w-5" />
              <span>开始推理 ({pendingFiles.length})</span>
            </>
          )}
        </button>

        <button
          onClick={clearFiles}
          disabled={uploadedFiles.length === 0}
          className="px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg
                     text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700
                     disabled:opacity-50 disabled:cursor-not-allowed"
        >
          清空
        </button>
      </div>
    </div>
  );
};

export default InferencePanel;
