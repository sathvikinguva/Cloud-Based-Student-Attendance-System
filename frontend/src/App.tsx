import React, { useState, useEffect } from 'react';
import AuthForm from './components/AuthForm';
import StudentDashboard from './components/StudentDashboard';
import TeacherDashboard from './components/TeacherDashboard';
import { User, UserRole, Student } from './types';
import { initializeLocalStorage } from './utils/initData';

const App = () => {
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [authType, setAuthType] = useState<'login' | 'signup'>('login');

  useEffect(() => {
    // Initialize localStorage with dummy data
    initializeLocalStorage();

    // Load the current user from localStorage if available
    const savedUser = localStorage.getItem('currentUser');
    if (savedUser) {
      setCurrentUser(JSON.parse(savedUser));
    }
  }, []);

  const handleAuth = (data: { email: string; password: string; name?: string; role?: UserRole }) => {
    if (authType === 'signup') {
      const users = JSON.parse(localStorage.getItem('users') || '[]');
      const existingUser = users.find((u: User) => u.email === data.email);

      if (existingUser) {
        alert('User already exists! Please use a different email.');
        return;
      }

      const newUser: User = {
        id: crypto.randomUUID(),
        name: data.name!,
        email: data.email,
        password: data.password,
        role: data.role!,
      };

      if (data.role === 'student') {
        (newUser as Student).isFirstLogin = true;
      }

      localStorage.setItem('users', JSON.stringify([...users, newUser]));
      localStorage.setItem('currentUser', JSON.stringify(newUser));
      setCurrentUser(newUser);
    } else {
      const users = JSON.parse(localStorage.getItem('users') || '[]');
      const user = users.find(
        (u: User) => u.email === data.email && u.password === data.password
      );

      if (user) {
        localStorage.setItem('currentUser', JSON.stringify(user));
        setCurrentUser(user);
      } else {
        alert('Invalid email or password! Please try again.');
      }
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('currentUser');
    setCurrentUser(null);
  };

  if (!currentUser) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50">
        <div className="container mx-auto px-4 py-8">
          <div className="max-w-md mx-auto">
            <h1 className="text-3xl font-bold text-center text-gray-800 mb-8">
              Student Attendance System
            </h1>
            <AuthForm type={authType} onSubmit={handleAuth} />
            <div className="text-center mt-6">
              <button
                onClick={() => setAuthType(authType === 'login' ? 'signup' : 'login')}
                className="text-indigo-600 hover:text-indigo-800 font-medium transition-colors"
              >
                {authType === 'login'
                  ? "Don't have an account? Sign up"
                  : 'Already have an account? Login'}
              </button>
            </div>
            {authType === 'login' && (
              <div className="mt-8 space-y-4 text-sm text-gray-600">
                <div className="bg-white p-4 rounded-lg shadow-sm">
                  <h3 className="font-semibold mb-2">Demo Accounts:</h3>
                  <div className="space-y-2">
                    <p><strong>Teacher:</strong> john@teacher.com / teacher123</p>
                    <p><strong>Student:</strong> alice@student.com / student123</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  return currentUser.role === 'student' ? (
    <StudentDashboard student={currentUser as Student} onLogout={handleLogout} />
  ) : (
    <TeacherDashboard teacher={currentUser} onLogout={handleLogout} />
  );
};

export default App;