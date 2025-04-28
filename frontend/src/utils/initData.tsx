import { User } from '../types';

// Dummy data for testing
const dummyUsers: User[] = [
  {
    id: '1',
    name: 'John Smith',
    email: 'john@teacher.com',
    password: 'teacher123',
    role: 'teacher',
  },
  {
    id: '2',
    name: 'Alice Johnson',
    email: 'alice@student.com',
    password: 'student123',
    role: 'student',
    isFirstLogin: true,
  },
];

export const initializeLocalStorage = () => {
  const isInitialized = localStorage.getItem('appInitialized');
  if (!isInitialized) {
    localStorage.clear();
    localStorage.setItem('users', JSON.stringify(dummyUsers));
    localStorage.setItem('attendanceRecords', JSON.stringify([]));
    localStorage.setItem('attendanceStatus', 'inactive');
    localStorage.setItem('appInitialized', 'true');
    console.log('Local storage initialized with dummy data.');
  }
};