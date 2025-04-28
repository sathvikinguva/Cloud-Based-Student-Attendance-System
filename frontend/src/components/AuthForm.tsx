import React, { useState } from 'react';
import { User, UserRole } from '../types';
import { Users, UserCircle2, Mail, Lock, User as UserIcon } from 'lucide-react';

interface AuthFormProps {
  type: 'login' | 'signup';
  onSubmit: (data: { email: string; password: string; name?: string; role?: UserRole }) => void;
}

export default function AuthForm({ type, onSubmit }: AuthFormProps) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [role, setRole] = useState<UserRole>('student');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (type === 'signup') {
      onSubmit({ email, password, name, role });
    } else {
      onSubmit({ email, password });
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-8">
      <div className="flex items-center justify-center mb-8">
        {type === 'login' ? (
          <div className="bg-indigo-100 p-3 rounded-full">
            <UserCircle2 size={40} className="text-indigo-600" />
          </div>
        ) : (
          <div className="bg-green-100 p-3 rounded-full">
            <Users size={40} className="text-green-600" />
          </div>
        )}
      </div>
      <h2 className="text-2xl font-bold text-center text-gray-800 mb-8">
        {type === 'login' ? 'Welcome Back!' : 'Create Your Account'}
      </h2>
      <form onSubmit={handleSubmit} className="space-y-6">
        {type === 'signup' && (
          <>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Full Name</label>
              <div className="relative">
                <UserIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="pl-10 w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="Enter your full name"
                  required
                />
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Role</label>
              <select
                value={role}
                onChange={(e) => setRole(e.target.value as UserRole)}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              >
                <option value="student">Student</option>
                <option value="teacher">Teacher</option>
              </select>
            </div>
          </>
        )}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Email Address</label>
          <div className="relative">
            <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="pl-10 w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              placeholder="Enter your email"
              required
            />
          </div>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
          <div className="relative">
            <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="pl-10 w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              placeholder="Enter your password"
              required
            />
          </div>
        </div>
        <button
          type="submit"
          className={`w-full py-3 px-4 rounded-lg text-white font-medium transition-colors ${
            type === 'login'
              ? 'bg-indigo-600 hover:bg-indigo-700'
              : 'bg-green-600 hover:bg-green-700'
          }`}
        >
          {type === 'login' ? 'Sign In' : 'Create Account'}
        </button>
      </form>
    </div>
  );
}