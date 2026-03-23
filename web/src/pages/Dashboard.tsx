/**
 * 主页/Dashboard
 */

import React from 'react';
import { Upload, Settings, History, BarChart3 } from 'lucide-react';
import UploadZone from '../components/UploadZone';
import InferencePanel from '../components/InferencePanel';
import ResultsPanel from '../components/ResultsPanel';

type Tab = 'upload' | 'history' | 'models';

export const Dashboard: React.FC = () => {
  const [activeTab, setActiveTab] = React.useState<Tab>('upload');

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* 头部 */}
      <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <BarChart3 className="h-8 w-8 text-blue-600" />
              <h1 className="text-xl font-bold text-gray-900 dark:text-gray-100">
                MONAI 3D 分割
              </h1>
            </div>
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-500">v1.0.0</span>
            </div>
          </div>
        </div>
      </header>

      {/* 标签导航 */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <nav className="flex space-x-8">
            {[
              { id: 'upload' as Tab, label: '推理', icon: Upload },
              { id: 'models' as Tab, label: '模型', icon: Settings },
              { id: 'history' as Tab, label: '历史', icon: History },
            ].map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className={`
                  flex items-center space-x-2 px-1 py-4 border-b-2 text-sm font-medium
                  ${activeTab === id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }
                `}
              >
                <Icon className="h-4 w-4" />
                <span>{label}</span>
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* 主内容 */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'upload' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* 左侧: 上传 + 配置 */}
            <div className="lg:col-span-2 space-y-6">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
                <h2 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-4">
                  上传图像
                </h2>
                <UploadZone />
              </div>
            </div>

            {/* 右侧: 配置 + 结果 */}
            <div className="space-y-6">
              <InferencePanel />
              <ResultsPanel />
            </div>
          </div>
        )}

        {activeTab === 'models' && (
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h2 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-4">
              模型管理
            </h2>
            <p className="text-gray-500">模型管理功能开发中...</p>
          </div>
        )}

        {activeTab === 'history' && (
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            <h2 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-4">
              历史记录
            </h2>
            <p className="text-gray-500">历史记录功能开发中...</p>
          </div>
        )}
      </main>
    </div>
  );
};

export default Dashboard;
