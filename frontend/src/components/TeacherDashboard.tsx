import React, { useState, useEffect } from 'react';
import { User, AttendanceRecord } from '../types';
import { LogOut, UserCircle, Users, CheckCircle2, XCircle, RefreshCw, RotateCcw } from 'lucide-react';

interface TeacherDashboardProps {
  teacher: User;
  onLogout: () => void;
}

export default function TeacherDashboard({ teacher, onLogout }: TeacherDashboardProps) {
  const [isAttendanceActive, setIsAttendanceActive] = useState(false);
  const [attendanceRecords, setAttendanceRecords] = useState<Array<AttendanceRecord>>([]);
  const [students, setStudents] = useState<User[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [resetting, setResetting] = useState(false);

  // Function to update attendance records from localStorage
  const loadAttendanceRecords = () => {
    const records = JSON.parse(localStorage.getItem('attendanceRecords') || '[]');
    setAttendanceRecords(records);
    
    // Check attendance status
    const status = localStorage.getItem('attendanceStatus');
    setIsAttendanceActive(status === 'active');
  };

  useEffect(() => {
    // Load students
    const users = JSON.parse(localStorage.getItem('users') || '[]');
    setStudents(users.filter((user: User) => user.role === 'student'));
    
    // Load attendance records
    loadAttendanceRecords();
    
    // Set up polling for updates
    const interval = setInterval(() => {
      loadAttendanceRecords();
    }, 3000); // Check every 3 seconds
    
    // Set up storage event listener for cross-tab updates
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'attendanceRecords' || e.key === 'attendanceStatus') {
        loadAttendanceRecords();
      } else if (e.key === 'users') {
        const updatedUsers = JSON.parse(e.newValue || '[]');
        setStudents(updatedUsers.filter((user: User) => user.role === 'student'));
      }
    };
    
    window.addEventListener('storage', handleStorageChange);
    
    // Cleanup
    return () => {
      clearInterval(interval);
      window.removeEventListener('storage', handleStorageChange);
    };
  }, []);

  const toggleAttendance = () => {
    const newStatus = !isAttendanceActive;
    setIsAttendanceActive(newStatus);
    localStorage.setItem('attendanceStatus', newStatus ? 'active' : 'inactive');

    if (!newStatus) {
      // Mark absent for students who didn't mark attendance
      const today = new Date().toLocaleDateString();
      
      // Get the latest records first
      const currentRecords = JSON.parse(localStorage.getItem('attendanceRecords') || '[]');
      
      const presentStudents = new Set(
        currentRecords
          .filter((record: AttendanceRecord) => record.date === today)
          .map((record: AttendanceRecord) => record.studentId)
      );

      const absentRecords: AttendanceRecord[] = students
        .filter(student => !presentStudents.has(student.id))
        .map(student => ({
          date: today,
          studentId: student.id,
          studentName: student.name,
          status: 'absent',
          timestamp: new Date().toISOString(),
        }));

      const updatedRecords = [...currentRecords, ...absentRecords];
      localStorage.setItem('attendanceRecords', JSON.stringify(updatedRecords));
      setAttendanceRecords(updatedRecords);
    }
  };

  const getStudentStatus = (studentId: string) => {
    const today = new Date().toLocaleDateString();
    
    // Find all records for today for this student
    const todayRecords = attendanceRecords.filter(
      record => record.studentId === studentId && record.date === today
    );
    
    // If there are multiple records, use the latest one
    if (todayRecords.length > 0) {
      // Sort by timestamp (newest first)
      todayRecords.sort((a, b) => 
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      );
      return todayRecords[0].status;
    }
    
    return 'pending';
  };

  // Regular refresh - just reloads data
  const refreshAttendance = () => {
    setRefreshing(true);
    loadAttendanceRecords();
    setTimeout(() => setRefreshing(false), 1000); // Show refresh animation for 1 second
  };

  // Reset attendance - removes all today's records
  const resetAttendance = () => {
    if (window.confirm('Are you sure you want to reset all attendance records for today? This will set all students to "pending" status.')) {
      setResetting(true);
      
      // Reset attendance statuses for today
      const today = new Date().toLocaleDateString();
      
      // Get all records from localStorage
      const allRecords = JSON.parse(localStorage.getItem('attendanceRecords') || '[]');
      
      // Filter out today's records
      const filteredRecords = allRecords.filter(
        (record: AttendanceRecord) => record.date !== today
      );
      
      // Save updated records
      localStorage.setItem('attendanceRecords', JSON.stringify(filteredRecords));
      setAttendanceRecords(filteredRecords);
      
      // Show reset animation
      setTimeout(() => {
        setResetting(false);
        // Reload attendance records after reset
        loadAttendanceRecords();
      }, 1000);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <div className="flex items-center">
              <UserCircle size={24} className="text-gray-500" />
              <span className="ml-2 font-semibold text-gray-900">{teacher.name}</span>
            </div>
            <button
              onClick={onLogout}
              className="flex items-center text-gray-500 hover:text-gray-700"
            >
              <LogOut size={20} />
              <span className="ml-2">Logout</span>
            </button>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          <div className="bg-white rounded-lg shadow">
            <div className="px-4 py-5 sm:p-6">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-xl font-semibold flex items-center">
                  <Users size={24} className="mr-2" />
                  Student Attendance
                </h2>
                <div className="flex space-x-2">
                  <button
                    onClick={refreshAttendance}
                    className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-md transition-colors flex items-center"
                    disabled={refreshing}
                  >
                    <RefreshCw size={18} className={`mr-1 ${refreshing ? 'animate-spin' : ''}`} />
                    Refresh
                  </button>
                  <button
                    onClick={resetAttendance}
                    className="bg-yellow-500 hover:bg-yellow-600 text-white px-4 py-2 rounded-md transition-colors flex items-center"
                    disabled={resetting}
                    title="Reset all attendance records for today"
                  >
                    <RotateCcw size={18} className={`mr-1 ${resetting ? 'animate-spin' : ''}`} />
                    Reset All
                  </button>
                  <button
                    onClick={toggleAttendance}
                    className={`${
                      isAttendanceActive
                        ? 'bg-red-500 hover:bg-red-600'
                        : 'bg-green-500 hover:bg-green-600'
                    } text-white px-4 py-2 rounded-md transition-colors`}
                  >
                    {isAttendanceActive ? 'Stop Attendance' : 'Start Attendance'}
                  </button>
                </div>
              </div>

              {isAttendanceActive && (
                <div className="mb-4 bg-yellow-50 border-l-4 border-yellow-400 p-4">
                  <div className="flex">
                    <div className="flex-shrink-0">
                      <svg className="h-5 w-5 text-yellow-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                        <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                      </svg>
                    </div>
                    <div className="ml-3">
                      <p className="text-sm text-yellow-700">
                        Attendance is currently active. Students can mark their attendance now.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Name
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Email
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Status
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {students.map((student) => {
                      const status = getStudentStatus(student.id);
                      return (
                        <tr key={student.id}>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                            {student.name}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {student.email}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span className={`flex items-center ${
                              status === 'present'
                                ? 'text-green-600'
                                : status === 'absent'
                                ? 'text-red-600'
                                : 'text-yellow-600'
                            }`}>
                              {status === 'present' ? (
                                <CheckCircle2 size={20} className="mr-1" />
                              ) : status === 'absent' ? (
                                <XCircle size={20} className="mr-1" />
                              ) : (
                                '⏳'
                              )}
                              {status.charAt(0).toUpperCase() + status.slice(1)}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}