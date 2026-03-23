/**
 * 结果展示面板
 */

import React from 'react';
import { Download, BarChart3, AlertCircle } from 'lucide-react';
import { useStore } from '../hooks/useStore';

export const ResultsPanel: React.FC = () => {
  const { uploadedFiles } = useStore();

  const completedFiles = uploadedFiles.filter((f) => f.status === 'done');
  const failedFiles = uploadedFiles.filter((f) => f.status === 'error');

  if (uploadedFiles.length === 0) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700 text-center">
        <BarChart3 className="mx-auto h-12 w-12 text-gray-400" />
        <p className="mt-2 text-sm text-gray-500">
          上传文件并完成推理后查看结果
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* 统计概览 */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
          <p className="text-2xl font-bold text-blue-600">{completedFiles.length}</p>
          <p className="text-xs text-gray-500">成功</p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
          <p className="text-2xl font-bold text-red-600">{failedFiles.length}</p>
          <p className="text-xs text-gray-500">失败</p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
          <p className="text-2xl font-bold text-gray-600">{uploadedFiles.length}</p>
          <p className="text-xs text-gray-500">总计</p>
        </div>
      </div>

      {/* 成功结果 */}
      {completedFiles.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
          <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
            <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100">
              推理结果
            </h3>
          </div>
          <div className="divide-y divide-gray-200 dark:divide-gray-700">
            {completedFiles.map((file) => (
              <div key={file.id} className="p-4">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                    {file.name}
                  </p>
                  <button className="text-blue-600 hover:text-blue-700 text-sm flex items-center space-x-1">
                    <Download className="h-4 w-4" />
                    <span>下载</span>
                  </button>
                </div>
                {file.result && (
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <p className="text-gray-500">Dice Score</p>
                      <p className="font-medium text-green-600">
                        {file.result.dice_score?.toFixed(4) || 'N/A'}
                      </p>
                    </div>
                    <div>
                      <p className="text-gray-500">体积 (cm³)</p>
                      <p className="font-medium">
                        {file.result.volume_cm3?.toFixed(2) || 'N/A'}
                      </p>
                    </div>
                    <div>
                      <p className="text-gray-500">处理时间</p>
                      <p className="font-medium">
                        {file.result.processing_time.toFixed(1)}s
                      </p>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 失败结果 */}
      {failedFiles.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-red-200 dark:border-red-800">
          <div className="px-4 py-3 border-b border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20">
            <h3 className="text-sm font-medium text-red-800 dark:text-red-200 flex items-center space-x-2">
              <AlertCircle className="h-4 w-4" />
              <span>失败列表</span>
            </h3>
          </div>
          <div className="divide-y divide-gray-200 dark:divide-gray-700">
            {failedFiles.map((file) => (
              <div key={file.id} className="p-4">
                <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                  {file.name}
                </p>
                <p className="text-sm text-red-600 mt-1">
                  {file.error || '未知错误'}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultsPanel;
