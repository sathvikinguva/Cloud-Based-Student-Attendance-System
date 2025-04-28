export type UserRole = 'teacher' | 'student';

export interface User {
  id: string;
  name: string;
  email: string;
  password: string;
  role: UserRole;
  isFirstLogin?: boolean;
}

export interface Student extends User {
  faceImage?: string;
  faceId?: string; // Add this property for face recognition
  isFirstLogin: boolean;
}

export interface AttendanceRecord {
  date: string;
  studentId: string;
  studentName: string;
  status: 'present' | 'absent';
  timestamp: string;
}